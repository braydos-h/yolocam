//! Local RTSP camera discovery.
//!
//! Three complementary mechanisms are used:
//!
//! * ONVIF WS-Discovery (UDP multicast, port 3702) finds ONVIF devices and
//!   their device-service URLs.
//! * SSDP / UPnP (UDP multicast, port 1900) finds devices and, when an HTTP
//!   description is reachable, their friendly name and model.
//! * A bounded TCP scan of common RTSP ports followed by RTSP DESCRIBE probes
//!   on well-known stream paths. This both confirms that a stream exists and
//!   reports the codec, or the authentication realm when login is required.
//!
//! Discovery never sends credentials, so users can pick a stream first and
//! enter the username/password in the existing settings form afterwards.

use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures_util::stream::{self, StreamExt};
use percent_encoding::percent_decode_str;
use serde::Serialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpStream, UdpSocket};
use tokio::sync::Mutex;
use tokio::time::timeout;
use url::Url;

const WS_DISCOVERY_ADDRESS: &str = "239.255.255.250:3702";
const SSDP_ADDRESS: &str = "239.255.255.250:1900";
const RTSP_PORTS: [u16; 4] = [554, 8554, 10554, 1554];
const MAX_SCAN_HOSTS: usize = 1024;
const MAX_RESULTS: usize = 64;
const MAX_RESPONSE_BYTES: usize = 64 * 1024;
const MAX_DESCRIPTION_BYTES: usize = 48 * 1024;
const HOST_CONCURRENCY: usize = 128;
const PATH_CONCURRENCY: usize = 6;
const MAX_VERIFIED_PER_ENDPOINT: usize = 4;
const MAX_AUTH_PER_ENDPOINT: usize = 4;

/// When a camera requires authentication for every request, the paths cannot
/// be verified, so only a few likely candidates are offered.
const AUTH_PROBE_PATHS: [&str; 5] = [
    "/",
    "/stream1",
    "/live",
    "/Streaming/Channels/101",
    "/cam/realmonitor?channel=1&subtype=0",
];

/// Well-known RTSP paths, ordered from the most common to vendor-specific
/// layouts. A camera usually answers only one or two of these.
const PROBE_PATHS: [&str; 24] = [
    "/",
    "/stream1",
    "/stream2",
    "/live",
    "/live/ch0",
    "/live/ch1",
    "/h264",
    "/h265",
    "/11",
    "/12",
    "/cam/realmonitor?channel=1&subtype=0",
    "/cam/realmonitor?channel=1&subtype=1",
    "/Streaming/Channels/101",
    "/Streaming/Channels/102",
    "/h264/ch1/main/av_stream",
    "/h264/ch1/sub/av_stream",
    "/onvif1",
    "/onvif2",
    "/axis-media/media.amp",
    "/media/video1",
    "/video1",
    "/main",
    "/sub",
    "/stream",
];

#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    #[error("a camera scan is already running")]
    Busy,
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Internal(String),
}

#[derive(Debug, Clone, Serialize)]
pub struct DiscoveryReport {
    /// Local network(s) or the explicit subnet that was scanned.
    pub scanned_subnets: Vec<String>,
    pub hosts_scanned: usize,
    pub duration_ms: u64,
    pub onvif_devices: Vec<OnvifDevice>,
    pub streams: Vec<DiscoveredStream>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OnvifDevice {
    pub host: String,
    pub service_url: Option<String>,
    pub name: Option<String>,
    pub hardware: Option<String>,
    pub location: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiscoveredStream {
    pub url: String,
    pub host: String,
    pub port: u16,
    pub path: String,
    pub source: DiscoverySource,
    /// True when the camera answered DESCRIBE with 200 OK and an SDP body.
    pub verified: bool,
    pub auth_required: bool,
    pub realm: Option<String>,
    pub codec: Option<String>,
    pub name: Option<String>,
    pub vendor: Option<String>,
    pub model: Option<String>,
    pub onvif_service: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoverySource {
    RtspScan,
    Onvif,
    Ssdp,
}

#[derive(Debug, Clone, Default)]
struct DeviceMeta {
    name: Option<String>,
    vendor: Option<String>,
    model: Option<String>,
    onvif_service: Option<String>,
    source: Option<DiscoverySource>,
}

#[derive(Debug, Clone)]
struct Probe {
    host: Ipv4Addr,
    port: u16,
    path: &'static str,
    auth_required: bool,
    realm: Option<String>,
    codec: Option<String>,
}

struct RtspResponse {
    status: u16,
    headers: String,
    body: String,
}

struct ScanPlan {
    hosts: Vec<Ipv4Addr>,
    subnets: Vec<String>,
    notes: Vec<String>,
}

/// Discover RTSP streams on private IPv4 networks.
///
/// `subnet` accepts a bare address (`192.168.1.50`, scanned as a single host)
/// or CIDR notation (`192.168.1.0/24`). When omitted, every /24 around a
/// locally routed private IPv4 address is scanned.
pub async fn discover(
    subnet: Option<&str>,
    timeout_ms: u64,
) -> Result<DiscoveryReport, DiscoveryError> {
    let _guard = discovery_lock()
        .try_lock()
        .map_err(|_| DiscoveryError::Busy)?;
    let started = Instant::now();
    let attempt_timeout = Duration::from_millis(timeout_ms.clamp(100, 3_000));

    let plan = build_scan_plan(subnet)?;
    let known: HashSet<Ipv4Addr> = plan.hosts.iter().copied().collect();

    // The subnet scan and the multicast discovery run in parallel; ONVIF or
    // SSDP devices on other subnets are probed afterwards.
    let scan = probe_hosts(plan.hosts.clone(), attempt_timeout);
    let multicast = async {
        tokio::join!(
            ws_discover(Duration::from_millis(1_500)),
            ssdp_discover(Duration::from_millis(1_200), attempt_timeout)
        )
    };
    let (mut probes, (ws_devices, ssdp_devices)) = tokio::join!(scan, multicast);

    let mut metadata: HashMap<Ipv4Addr, DeviceMeta> = HashMap::new();
    let mut onvif_devices = Vec::new();
    let mut extra_hosts = Vec::new();

    for device in ws_devices {
        let meta = metadata.entry(device.host).or_default();
        merge_option(&mut meta.name, device.name.clone());
        merge_option(&mut meta.model, device.hardware.clone());
        merge_option(&mut meta.onvif_service, device.service_url.clone());
        meta.source = Some(DiscoverySource::Onvif);
        if !known.contains(&device.host) && !extra_hosts.contains(&device.host) {
            extra_hosts.push(device.host);
        }
        onvif_devices.push(OnvifDevice {
            host: device.host.to_string(),
            service_url: device.service_url,
            name: device.name,
            hardware: device.hardware,
            location: device.location,
        });
    }

    for device in ssdp_devices {
        let meta = metadata.entry(device.host).or_default();
        merge_option(&mut meta.name, device.name);
        merge_option(&mut meta.vendor, device.vendor);
        merge_option(&mut meta.model, device.model);
        if meta.source.is_none() {
            meta.source = Some(DiscoverySource::Ssdp);
        }
        if !known.contains(&device.host) && !extra_hosts.contains(&device.host) {
            extra_hosts.push(device.host);
        }
    }

    if !extra_hosts.is_empty() {
        probes.extend(probe_hosts(extra_hosts, attempt_timeout).await);
    }

    let streams = build_streams(probes, &metadata);
    let hosts_scanned = plan
        .hosts
        .len()
        .saturating_add(metadata.keys().filter(|host| !known.contains(host)).count());

    let mut notes = plan.notes;
    if streams.is_empty() {
        notes.push(
            "No RTSP stream answered. If the camera is on another subnet, enter that subnet and scan again."
                .to_string(),
        );
    }
    if streams.iter().any(|stream| stream.auth_required) {
        notes.push(
            "Some cameras require a login. Select a stream, then enter the camera username and password before saving."
                .to_string(),
        );
    }
    if !onvif_devices.is_empty() {
        notes.push(format!(
            "{} ONVIF device(s) answered WS-Discovery.",
            onvif_devices.len()
        ));
    }

    Ok(DiscoveryReport {
        scanned_subnets: plan.subnets,
        hosts_scanned,
        duration_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        onvif_devices,
        streams,
        notes,
    })
}

fn discovery_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn build_scan_plan(subnet: Option<&str>) -> Result<ScanPlan, DiscoveryError> {
    let mut notes = Vec::new();
    if let Some(text) = subnet.map(str::trim).filter(|text| !text.is_empty()) {
        let parsed = parse_subnet(text)?;
        return Ok(ScanPlan {
            hosts: parsed.hosts,
            subnets: vec![parsed.label],
            notes,
        });
    }

    let local = local_ipv4_addresses();
    if local.is_empty() {
        return Err(DiscoveryError::Invalid(
            "cannot determine the local network; enter a subnet such as 192.168.1.0/24".to_string(),
        ));
    }

    let mut hosts = Vec::new();
    let mut seen = HashSet::new();
    let mut subnets = Vec::new();
    for address in local {
        let parsed = parse_subnet(&format!("{address}/24"))?;
        if seen.insert(parsed.label.clone()) {
            subnets.push(parsed.label);
            hosts.extend(parsed.hosts);
        }
    }
    if hosts.is_empty() {
        return Err(DiscoveryError::Invalid(
            "the local network has no scannable hosts; enter a subnet such as 192.168.1.0/24"
                .to_string(),
        ));
    }
    if hosts.len() > MAX_SCAN_HOSTS {
        hosts.truncate(MAX_SCAN_HOSTS);
        notes.push(format!(
            "Scan limited to {MAX_SCAN_HOSTS} hosts. Enter a smaller subnet to scan a specific range."
        ));
    }
    Ok(ScanPlan {
        hosts,
        subnets,
        notes,
    })
}

struct ParsedSubnet {
    label: String,
    hosts: Vec<Ipv4Addr>,
}

fn parse_subnet(text: &str) -> Result<ParsedSubnet, DiscoveryError> {
    let text = text.trim();
    let (address_text, prefix) = match text.split_once('/') {
        Some((address, prefix)) => {
            let prefix = prefix.trim().parse::<u8>().map_err(|_| {
                DiscoveryError::Invalid("subnet prefix must be between 0 and 32".into())
            })?;
            (address.trim(), prefix)
        }
        None => (text, 32),
    };
    let address = address_text.parse::<Ipv4Addr>().map_err(|_| {
        DiscoveryError::Invalid(format!(
            "\"{text}\" is not an IPv4 address or CIDR subnet (example: 192.168.1.0/24)"
        ))
    })?;
    if prefix > 32 {
        return Err(DiscoveryError::Invalid(
            "subnet prefix must be between 0 and 32".into(),
        ));
    }
    if !is_private_v4(address) {
        return Err(DiscoveryError::Invalid(
            "only private IPv4 networks can be scanned (10/8, 172.16/12, 192.168/16, 169.254/16, 100.64/10)"
                .into(),
        ));
    }
    if prefix < 22 {
        return Err(DiscoveryError::Invalid(
            "subnet is too large; use /22 or smaller (for example 192.168.1.0/24)".into(),
        ));
    }

    let mask = if prefix == 0 {
        0
    } else {
        u32::MAX << (32 - prefix)
    };
    let network = Ipv4Addr::from(u32::from(address) & mask);
    let broadcast = Ipv4Addr::from(u32::from(network) | !mask);

    let (first, last) = if prefix >= 31 {
        (u32::from(network), u32::from(broadcast))
    } else {
        (u32::from(network) + 1, u32::from(broadcast) - 1)
    };
    if last < first {
        return Err(DiscoveryError::Invalid(
            "subnet contains no usable hosts".into(),
        ));
    }
    let count = last - first + 1;
    if count as usize > MAX_SCAN_HOSTS {
        return Err(DiscoveryError::Invalid(format!(
            "subnet contains {count} hosts; scan at most {MAX_SCAN_HOSTS} at a time"
        )));
    }
    let hosts = (first..=last).map(Ipv4Addr::from).collect();
    Ok(ParsedSubnet {
        label: format!("{network}/{prefix}"),
        hosts,
    })
}

/// Local IPv4 addresses of interfaces that are up. Scanning every interface
/// matters on machines with Ethernet and Wi-Fi attached to different networks.
#[cfg(unix)]
fn local_ipv4_addresses() -> Vec<Ipv4Addr> {
    let mut addresses: Vec<Ipv4Addr> = Vec::new();
    let mut interfaces: *mut libc::ifaddrs = std::ptr::null_mut();
    // getifaddrs returns a linked list owned by libc; freeifaddrs releases it.
    if unsafe { libc::getifaddrs(&mut interfaces) } != 0 {
        return route_local_ipv4_addresses();
    }
    let mut cursor = interfaces;
    while !cursor.is_null() {
        let entry = unsafe { &*cursor };
        let interface_name = unsafe { std::ffi::CStr::from_ptr(entry.ifa_name) }.to_string_lossy();
        if !entry.ifa_addr.is_null()
            && !is_virtual_interface(&interface_name)
            && entry.ifa_flags & (libc::IFF_UP as u32) != 0
            && entry.ifa_flags & (libc::IFF_LOOPBACK as u32) == 0
            && unsafe { (*entry.ifa_addr).sa_family as i32 } == libc::AF_INET
        {
            let sockaddr = unsafe { &*(entry.ifa_addr as *const libc::sockaddr_in) };
            let address = Ipv4Addr::from(sockaddr.sin_addr.s_addr.to_ne_bytes());
            if !address.is_loopback()
                && !address.is_unspecified()
                && is_private_v4(address)
                && !addresses.contains(&address)
            {
                addresses.push(address);
            }
        }
        cursor = entry.ifa_next;
    }
    unsafe { libc::freeifaddrs(interfaces) };
    if addresses.is_empty() {
        route_local_ipv4_addresses()
    } else {
        addresses
    }
}

#[cfg(not(unix))]
fn local_ipv4_addresses() -> Vec<Ipv4Addr> {
    route_local_ipv4_addresses()
}

/// Container and virtual-machine bridges rarely host cameras and would add
/// hundreds of unreachable hosts to every scan.
#[cfg(unix)]
fn is_virtual_interface(name: &str) -> bool {
    ["docker", "veth", "virbr", "br-", "lxc", "cni", "flannel"]
        .iter()
        .any(|prefix| name.starts_with(prefix))
}

/// Fallback for platforms without getifaddrs. A UDP socket connect performs a
/// route lookup but sends no packets; the chosen source address identifies
/// the interface used for each private range.
fn route_local_ipv4_addresses() -> Vec<Ipv4Addr> {
    const DESTINATIONS: [&str; 6] = [
        "192.168.254.254",
        "10.254.254.254",
        "172.31.254.254",
        "169.254.254.254",
        "100.127.254.254",
        "8.8.8.8",
    ];
    let mut addresses = Vec::new();
    for destination in DESTINATIONS {
        let Ok(socket) = std::net::UdpSocket::bind("0.0.0.0:0") else {
            continue;
        };
        if socket.connect((destination, 9)).is_err() {
            continue;
        }
        let Ok(local) = socket.local_addr() else {
            continue;
        };
        let IpAddr::V4(address) = local.ip() else {
            continue;
        };
        if address.is_loopback() || address.is_unspecified() || !is_private_v4(address) {
            continue;
        }
        if !addresses.contains(&address) {
            addresses.push(address);
        }
    }
    addresses
}

fn is_private_v4(address: Ipv4Addr) -> bool {
    let octets = address.octets();
    octets[0] == 10
        || (octets[0] == 172 && (16..=31).contains(&octets[1]))
        || (octets[0] == 192 && octets[1] == 168)
        || (octets[0] == 169 && octets[1] == 254)
        || (octets[0] == 100 && (64..=127).contains(&octets[1]))
}

async fn probe_hosts(hosts: Vec<Ipv4Addr>, attempt_timeout: Duration) -> Vec<Probe> {
    let endpoints: Vec<(Ipv4Addr, u16)> = hosts
        .into_iter()
        .flat_map(|host| RTSP_PORTS.map(|port| (host, port)))
        .collect();
    let futures = endpoints
        .into_iter()
        .map(|(host, port)| probe_endpoint(host, port, attempt_timeout));
    let results = stream::iter(futures)
        .buffer_unordered(HOST_CONCURRENCY)
        .collect::<Vec<_>>()
        .await;
    results.into_iter().flatten().collect()
}

async fn probe_endpoint(host: Ipv4Addr, port: u16, attempt_timeout: Duration) -> Vec<Probe> {
    // Cheap sanity check: only probe paths when the service actually speaks
    // RTSP. This keeps unrelated TCP services out of the results.
    let options = match rtsp_request(host, port, "OPTIONS", "/", attempt_timeout).await {
        Some(response) => response,
        None => match rtsp_request(host, port, "OPTIONS", "*", attempt_timeout).await {
            Some(response) => response,
            None => return Vec::new(),
        },
    };

    let paths: Vec<&'static str> = if options.status == 401 {
        AUTH_PROBE_PATHS.to_vec()
    } else {
        PROBE_PATHS.to_vec()
    };
    let mut tasks = Vec::with_capacity(paths.len());
    for path in paths {
        tasks.push(probe_path(host, port, path, attempt_timeout));
    }
    let results = stream::iter(tasks)
        .buffer_unordered(PATH_CONCURRENCY)
        .filter_map(|result| async move { result })
        .collect::<Vec<_>>()
        .await;

    let mut verified = Vec::new();
    let mut auth = Vec::new();
    for result in results {
        if result.auth_required {
            auth.push(result);
        } else {
            verified.push(result);
        }
    }
    verified.sort_by_key(|probe| path_rank(probe.path));
    auth.sort_by_key(|probe| path_rank(probe.path));

    if verified.is_empty() {
        auth.truncate(MAX_AUTH_PER_ENDPOINT);
        auth
    } else {
        verified.truncate(MAX_VERIFIED_PER_ENDPOINT);
        verified
    }
}

async fn probe_path(
    host: Ipv4Addr,
    port: u16,
    path: &'static str,
    attempt_timeout: Duration,
) -> Option<Probe> {
    let response = rtsp_request(host, port, "DESCRIBE", path, attempt_timeout).await?;
    if response.status != 200 && response.status != 401 {
        return None;
    }
    let auth_required = response.status == 401;
    let realm = if auth_required {
        header_value(&response.headers, "www-authenticate").and_then(|value| parse_realm(&value))
    } else {
        None
    };
    let codec = if response.status == 200 {
        sdp_codec(&response.body)
    } else {
        None
    };
    Some(Probe {
        host,
        port,
        path,
        auth_required,
        realm,
        codec,
    })
}

async fn rtsp_request(
    host: Ipv4Addr,
    port: u16,
    method: &str,
    path: &str,
    attempt_timeout: Duration,
) -> Option<RtspResponse> {
    let address = SocketAddr::from((host, port));
    let mut stream = match timeout(attempt_timeout, TcpStream::connect(address)).await {
        Ok(Ok(stream)) => stream,
        _ => return None,
    };
    let url = if path == "*" {
        "*".to_string()
    } else {
        format!("rtsp://{host}:{port}{path}")
    };
    let extra = if method == "DESCRIBE" {
        "Accept: application/sdp\r\n"
    } else {
        ""
    };
    let request = format!(
        "{method} {url} RTSP/1.0\r\nCSeq: 1\r\nUser-Agent: YOLOCam/{}\r\n{extra}\r\n",
        env!("CARGO_PKG_VERSION")
    );
    match timeout(attempt_timeout, stream.write_all(request.as_bytes())).await {
        Ok(Ok(())) => {}
        _ => return None,
    }
    let raw = timeout(attempt_timeout, read_rtsp_message(&mut stream))
        .await
        .ok()
        .and_then(Result::ok)?;
    parse_rtsp_response(&raw)
}

async fn read_rtsp_message(stream: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut buffer = Vec::with_capacity(2_048);
    let mut chunk = [0u8; 4_096];
    let header_end = loop {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Ok(buffer);
        }
        buffer.extend_from_slice(&chunk[..read]);
        if buffer.len() > MAX_RESPONSE_BYTES {
            return Ok(buffer);
        }
        if let Some(position) = find_subslice(&buffer, b"\r\n\r\n") {
            break position;
        }
    };
    let head = String::from_utf8_lossy(&buffer[..header_end]).to_string();
    let content_length = header_value(&head, "content-length")
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0)
        .min(MAX_RESPONSE_BYTES);
    let total = header_end + 4 + content_length;
    while buffer.len() < total {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if buffer.len() > MAX_RESPONSE_BYTES + 4_096 {
            break;
        }
    }
    Ok(buffer)
}

fn parse_rtsp_response(raw: &[u8]) -> Option<RtspResponse> {
    let text = String::from_utf8_lossy(raw);
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    let status_line = head.lines().next()?;
    if !status_line.starts_with("RTSP/") {
        return None;
    }
    let status = status_line.split_whitespace().nth(1)?.parse::<u16>().ok()?;
    Some(RtspResponse {
        status,
        headers: head.to_string(),
        body: body.to_string(),
    })
}

fn header_value(headers: &str, name: &str) -> Option<String> {
    headers.lines().skip(1).find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.trim()
            .eq_ignore_ascii_case(name)
            .then(|| value.trim().to_string())
    })
}

fn parse_realm(header: &str) -> Option<String> {
    let lower = header.to_ascii_lowercase();
    let start = lower.find("realm=")? + "realm=".len();
    let rest = &header[start..];
    let value = if let Some(quoted) = rest.strip_prefix('"') {
        quoted.split('"').next().unwrap_or(quoted)
    } else {
        rest.split([',', ';']).next().unwrap_or(rest).trim()
    };
    (!value.is_empty()).then(|| value.to_string())
}

fn sdp_codec(body: &str) -> Option<String> {
    for line in body.lines() {
        let Some(rest) = line.trim().strip_prefix("a=rtpmap:") else {
            continue;
        };
        let Some((_, encoding)) = rest.split_once(' ') else {
            continue;
        };
        let name = encoding.split('/').next().unwrap_or(encoding).trim();
        if name.is_empty() {
            continue;
        }
        let label = match name.to_ascii_uppercase().as_str() {
            "H264" => "H.264",
            "H265" | "HEVC" => "H.265",
            "MP4V-ES" => "MPEG-4",
            "JPEG" | "MJPEG" => "MJPEG",
            "MPEG4-GENERIC" => "AAC",
            _ => name,
        };
        return Some(label.to_string());
    }
    None
}

fn path_rank(path: &str) -> usize {
    PROBE_PATHS
        .iter()
        .position(|candidate| *candidate == path)
        .unwrap_or(PROBE_PATHS.len())
}

fn build_streams(
    probes: Vec<Probe>,
    metadata: &HashMap<Ipv4Addr, DeviceMeta>,
) -> Vec<DiscoveredStream> {
    let mut seen = HashSet::new();
    let mut streams = Vec::new();
    for probe in probes {
        let url = format!("rtsp://{}:{}{}", probe.host, probe.port, probe.path);
        if !seen.insert(url.clone()) {
            continue;
        }
        let meta = metadata.get(&probe.host);
        let source = meta
            .and_then(|meta| meta.source)
            .unwrap_or(DiscoverySource::RtspScan);
        streams.push(DiscoveredStream {
            url,
            host: probe.host.to_string(),
            port: probe.port,
            path: probe.path.to_string(),
            source,
            verified: !probe.auth_required,
            auth_required: probe.auth_required,
            realm: probe.realm,
            codec: probe.codec,
            name: meta.and_then(|meta| meta.name.clone()),
            vendor: meta.and_then(|meta| meta.vendor.clone()),
            model: meta.and_then(|meta| meta.model.clone()),
            onvif_service: meta.and_then(|meta| meta.onvif_service.clone()),
        });
    }
    streams.sort_by(|a, b| {
        b.verified
            .cmp(&a.verified)
            .then_with(|| a.host.cmp(&b.host))
            .then_with(|| a.port.cmp(&b.port))
            .then_with(|| path_rank(&a.path).cmp(&path_rank(&b.path)))
            .then_with(|| a.url.cmp(&b.url))
    });
    streams.truncate(MAX_RESULTS);
    streams
}

#[derive(Debug)]
struct WsDevice {
    host: Ipv4Addr,
    service_url: Option<String>,
    name: Option<String>,
    hardware: Option<String>,
    location: Option<String>,
}

async fn ws_discover(duration: Duration) -> Vec<WsDevice> {
    let Ok(socket) = UdpSocket::bind("0.0.0.0:0").await else {
        return Vec::new();
    };
    let Ok(target) = WS_DISCOVERY_ADDRESS.parse::<SocketAddr>() else {
        return Vec::new();
    };
    let probe = ws_probe_message();
    for _ in 0..2 {
        let _ = socket.send_to(probe.as_bytes(), target).await;
    }

    let deadline = Instant::now() + duration;
    let mut buffer = [0u8; 8_192];
    let mut devices: Vec<WsDevice> = Vec::new();
    let mut seen = HashSet::new();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        let Ok(Ok((read, source))) = timeout(remaining, socket.recv_from(&mut buffer)).await else {
            break;
        };
        let text = String::from_utf8_lossy(&buffer[..read]);
        let Some(device) = parse_ws_response(&text, source) else {
            continue;
        };
        if seen.insert((device.host, device.service_url.clone())) {
            devices.push(device);
        }
        if devices.len() >= MAX_RESULTS {
            break;
        }
    }
    devices
}

fn ws_probe_message() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let id = nanos ^ (u128::from(counter) << 96);
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
 xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
 xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
 <e:Header>
  <w:MessageID>urn:uuid:{id:032x}</w:MessageID>
  <w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
  <w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
 </e:Header>
 <e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>
</e:Envelope>"#
    )
}

fn parse_ws_response(xml: &str, source: SocketAddr) -> Option<WsDevice> {
    let scopes = extract_tag(xml, "Scopes");
    let xaddrs = extract_tag(xml, "XAddrs");
    let service_url = xaddrs
        .as_deref()
        .and_then(|value| value.split_whitespace().find(|url| url.starts_with("http")))
        .map(str::to_string);
    let host = service_url
        .as_deref()
        .and_then(url_ipv4)
        .or_else(|| match source.ip() {
            IpAddr::V4(address) if is_private_v4(address) => Some(address),
            _ => None,
        })?;
    if !is_private_v4(host) {
        return None;
    }
    let name = scopes
        .as_deref()
        .and_then(|value| scope_value(value, "name"));
    let hardware = scopes
        .as_deref()
        .and_then(|value| scope_value(value, "hardware"));
    let location = scopes
        .as_deref()
        .and_then(|value| scope_value(value, "location"));
    Some(WsDevice {
        host,
        service_url,
        name,
        hardware,
        location,
    })
}

fn scope_value(scopes: &str, key: &str) -> Option<String> {
    scopes.split_whitespace().find_map(|scope| {
        let prefix = format!("onvif://www.onvif.org/{key}/");
        let value = scope.strip_prefix(&prefix)?;
        percent_decode_str(value)
            .decode_utf8()
            .ok()
            .map(|value| value.into_owned())
            .filter(|value| !value.trim().is_empty())
    })
}

#[derive(Debug)]
struct SsdpDevice {
    host: Ipv4Addr,
    name: Option<String>,
    vendor: Option<String>,
    model: Option<String>,
}

async fn ssdp_discover(duration: Duration, fetch_timeout: Duration) -> Vec<SsdpDevice> {
    let Ok(socket) = UdpSocket::bind("0.0.0.0:0").await else {
        return Vec::new();
    };
    let Ok(target) = SSDP_ADDRESS.parse::<SocketAddr>() else {
        return Vec::new();
    };
    let request = format!(
        "M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: \"ssdp:discover\"\r\nMX: 1\r\nST: ssdp:all\r\nUSER-AGENT: YOLOCam/{}\r\n\r\n",
        env!("CARGO_PKG_VERSION")
    );
    for _ in 0..2 {
        let _ = socket.send_to(request.as_bytes(), target).await;
    }

    let deadline = Instant::now() + duration;
    let mut buffer = [0u8; 4_096];
    let mut locations: Vec<(Ipv4Addr, String)> = Vec::new();
    let mut seen = HashSet::new();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        let Ok(Ok((read, source))) = timeout(remaining, socket.recv_from(&mut buffer)).await else {
            break;
        };
        let IpAddr::V4(source_host) = source.ip() else {
            continue;
        };
        if !is_private_v4(source_host) {
            continue;
        }
        let text = String::from_utf8_lossy(&buffer[..read]);
        let Some(location) = ssdp_location(&text) else {
            continue;
        };
        if seen.insert(location.clone()) {
            locations.push((source_host, location));
        }
        if locations.len() >= 24 {
            break;
        }
    }

    let tasks = locations
        .into_iter()
        .map(|(source_host, location)| async move {
            let described = fetch_description(&location, fetch_timeout).await;
            let host = described
                .as_ref()
                .and_then(|description| description.host)
                .or_else(|| url_ipv4(&location))
                .unwrap_or(source_host);
            (host, described)
        });
    let results = stream::iter(tasks)
        .buffer_unordered(8)
        .collect::<Vec<_>>()
        .await;

    let mut devices = Vec::new();
    let mut seen_hosts = HashSet::new();
    for (host, described) in results {
        if !is_private_v4(host) || !seen_hosts.insert(host) {
            continue;
        }
        devices.push(SsdpDevice {
            host,
            name: described
                .as_ref()
                .and_then(|description| description.name.clone()),
            vendor: described
                .as_ref()
                .and_then(|description| description.vendor.clone()),
            model: described
                .as_ref()
                .and_then(|description| description.model.clone()),
        });
    }
    devices
}

fn ssdp_location(response: &str) -> Option<String> {
    response.lines().skip(1).find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.trim()
            .eq_ignore_ascii_case("location")
            .then(|| value.trim().to_string())
            .filter(|value| value.starts_with("http://"))
    })
}

struct Description {
    host: Option<Ipv4Addr>,
    name: Option<String>,
    vendor: Option<String>,
    model: Option<String>,
}

async fn fetch_description(location: &str, attempt_timeout: Duration) -> Option<Description> {
    let url = Url::parse(location).ok()?;
    if url.scheme() != "http" {
        return None;
    }
    let host_text = url.host_str()?;
    let address = SocketAddr::from((
        host_text.parse::<Ipv4Addr>().ok()?,
        url.port_or_known_default().unwrap_or(80),
    ));
    let deadline = Instant::now() + attempt_timeout;
    let remaining = || deadline.saturating_duration_since(Instant::now());
    let mut stream = timeout(remaining(), TcpStream::connect(address))
        .await
        .ok()?
        .ok()?;
    let mut path = url.path().to_string();
    if path.is_empty() {
        path.push('/');
    }
    if let Some(query) = url.query() {
        path.push('?');
        path.push_str(query);
    }
    let host_header = match url.port() {
        Some(port) => format!("{host_text}:{port}"),
        None => host_text.to_string(),
    };
    let request = format!(
        "GET {path} HTTP/1.0\r\nHost: {host_header}\r\nConnection: close\r\nUser-Agent: YOLOCam/{}\r\n\r\n",
        env!("CARGO_PKG_VERSION")
    );
    timeout(remaining(), stream.write_all(request.as_bytes()))
        .await
        .ok()?
        .ok()?;

    let mut body = Vec::new();
    let mut chunk = [0u8; 4_096];
    loop {
        let left = remaining();
        if left.is_zero() {
            break;
        }
        let read = timeout(left, stream.read(&mut chunk)).await.ok()?.ok()?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..read]);
        if body.len() > MAX_DESCRIPTION_BYTES {
            break;
        }
    }
    let text = String::from_utf8_lossy(&body);
    let xml = text
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .unwrap_or(&text);
    let IpAddr::V4(host) = address.ip() else {
        return None;
    };
    Some(Description {
        host: Some(host),
        name: extract_tag(xml, "friendlyName"),
        vendor: extract_tag(xml, "manufacturer"),
        model: extract_tag(xml, "modelName"),
    })
}

fn url_ipv4(value: &str) -> Option<Ipv4Addr> {
    let url = Url::parse(value).ok()?;
    url.host_str()?.parse::<Ipv4Addr>().ok()
}

fn merge_option(target: &mut Option<String>, value: Option<String>) {
    if target.is_none() {
        if let Some(value) = value {
            if !value.trim().is_empty() {
                *target = Some(value);
            }
        }
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Extract the text content of an XML element, tolerating namespace prefixes
/// and attributes. Discovery responses originate from cameras, so this is
/// deliberately small and never evaluates anything from the response.
fn extract_tag(xml: &str, tag: &str) -> Option<String> {
    let mut offset = 0;
    while let Some(relative_open) = xml[offset..].find('<') {
        let open = offset + relative_open;
        let Some(relative_end) = xml[open..].find('>') else {
            return None;
        };
        let end = open + relative_end;
        let raw_name = xml[open + 1..end].trim();
        if raw_name.starts_with('/') || raw_name.starts_with('?') || raw_name.starts_with('!') {
            offset = end + 1;
            continue;
        }
        let name = raw_name
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_end_matches('/');
        let bare = name.rsplit(':').next().unwrap_or(name);
        if bare == tag && !raw_name.ends_with('/') {
            let close_with_prefix = format!("</{name}>");
            let close_plain = format!("</{tag}>");
            if let Some(relative_close) = xml[end + 1..]
                .find(&close_with_prefix)
                .or_else(|| xml[end + 1..].find(&close_plain))
            {
                let content = &xml[end + 1..end + 1 + relative_close];
                return Some(decode_xml_entities(content));
            }
        }
        offset = end + 1;
    }
    None
}

fn decode_xml_entities(value: &str) -> String {
    value
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subnet_parser_makes_usable_host_range() {
        let parsed = parse_subnet("192.168.1.0/24").unwrap();
        assert_eq!(parsed.hosts.len(), 254);
        assert_eq!(parsed.hosts.first(), Some(&Ipv4Addr::new(192, 168, 1, 1)));
        assert_eq!(parsed.hosts.last(), Some(&Ipv4Addr::new(192, 168, 1, 254)));
        assert_eq!(parsed.label, "192.168.1.0/24");
    }

    #[test]
    fn subnet_parser_supports_single_host_and_host_bits() {
        let single = parse_subnet("10.0.0.7").unwrap();
        assert_eq!(single.hosts, vec![Ipv4Addr::new(10, 0, 0, 7)]);
        let masked = parse_subnet("192.168.1.77/30").unwrap();
        assert_eq!(masked.label, "192.168.1.76/30");
        assert_eq!(masked.hosts.len(), 2);
    }

    #[test]
    fn subnet_parser_rejects_public_and_huge_ranges() {
        assert!(parse_subnet("8.8.8.0/24").is_err());
        assert!(parse_subnet("192.168.0.0/8").is_err());
        assert!(parse_subnet("not-a-subnet").is_err());
    }

    #[test]
    fn parses_rtsp_ok_and_unauthorized() {
        let ok = b"RTSP/1.0 200 OK\r\nCSeq: 1\r\nContent-Length: 31\r\n\r\nv=0\r\na=rtpmap:96 H264/90000\r\n";
        let response = parse_rtsp_response(ok).unwrap();
        assert_eq!(response.status, 200);
        assert_eq!(sdp_codec(&response.body).as_deref(), Some("H.264"));

        let unauthorized = b"RTSP/1.0 401 Unauthorized\r\nCSeq: 1\r\nWWW-Authenticate: Digest realm=\"IP Camera\", nonce=\"abc\"\r\n\r\n";
        let response = parse_rtsp_response(unauthorized).unwrap();
        assert_eq!(response.status, 401);
        let realm = header_value(&response.headers, "www-authenticate")
            .and_then(|value| parse_realm(&value));
        assert_eq!(realm.as_deref(), Some("IP Camera"));
    }

    #[test]
    fn rejects_non_rtsp_responses() {
        assert!(parse_rtsp_response(b"HTTP/1.1 200 OK\r\n\r\n").is_none());
    }

    #[test]
    fn parses_ws_discovery_response() {
        let xml = r#"<?xml version="1.0"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery">
 <e:Body><d:ProbeMatches><d:ProbeMatch>
  <d:Scopes>onvif://www.onvif.org/name/Front%20Door onvif://www.onvif.org/hardware/Model%20X</d:Scopes>
  <d:XAddrs>http://192.168.1.50/onvif/device_service</d:XAddrs>
 </d:ProbeMatch></d:ProbeMatches></e:Body>
</e:Envelope>"#;
        let source: SocketAddr = "192.168.1.50:3702".parse().unwrap();
        let device = parse_ws_response(xml, source).unwrap();
        assert_eq!(device.host, Ipv4Addr::new(192, 168, 1, 50));
        assert_eq!(device.name.as_deref(), Some("Front Door"));
        assert_eq!(device.hardware.as_deref(), Some("Model X"));
        assert_eq!(
            device.service_url.as_deref(),
            Some("http://192.168.1.50/onvif/device_service")
        );
    }

    #[test]
    fn extracts_ssdp_description_fields() {
        let xml = "<root><device><friendlyName>Driveway &amp; Gate</friendlyName><manufacturer>Acme</manufacturer><modelName>Cam 2000</modelName></device></root>";
        assert_eq!(
            extract_tag(xml, "friendlyName").as_deref(),
            Some("Driveway & Gate")
        );
        assert_eq!(extract_tag(xml, "manufacturer").as_deref(), Some("Acme"));
        assert_eq!(extract_tag(xml, "modelName").as_deref(), Some("Cam 2000"));
        assert!(extract_tag(xml, "missing").is_none());
    }

    #[test]
    fn ssdp_location_is_parsed_case_insensitively() {
        let response = "HTTP/1.1 200 OK\r\nCACHE-CONTROL: max-age=1800\r\nlocation: http://192.168.1.2:49152/description.xml\r\nST: upnp:rootdevice\r\n\r\n";
        assert_eq!(
            ssdp_location(response).as_deref(),
            Some("http://192.168.1.2:49152/description.xml")
        );
    }

    #[tokio::test]
    async fn discovery_rejects_public_subnets_before_scanning() {
        let result = discover(Some("8.8.8.0/24"), 100).await;
        assert!(matches!(result, Err(DiscoveryError::Invalid(_))));
    }

    #[test]
    fn stream_results_are_deduplicated_and_sorted() {
        let probes = vec![
            Probe {
                host: Ipv4Addr::new(192, 168, 1, 9),
                port: 554,
                path: "/stream1",
                auth_required: false,
                realm: None,
                codec: Some("H.264".into()),
            },
            Probe {
                host: Ipv4Addr::new(192, 168, 1, 2),
                port: 554,
                path: "/",
                auth_required: true,
                realm: Some("Camera".into()),
                codec: None,
            },
            Probe {
                host: Ipv4Addr::new(192, 168, 1, 2),
                port: 554,
                path: "/",
                auth_required: true,
                realm: Some("Camera".into()),
                codec: None,
            },
        ];
        let streams = build_streams(probes, &HashMap::new());
        assert_eq!(streams.len(), 2);
        assert!(streams[0].verified);
        assert_eq!(streams[0].host, "192.168.1.9");
        assert_eq!(streams[1].realm.as_deref(), Some("Camera"));
    }
}
