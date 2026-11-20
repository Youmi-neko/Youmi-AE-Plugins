//! Free video memory of the adapter, as the OS sees it (other applications included).
//!
//! AE, its GPU renderer and other tools (ML tools are common on artists' machines) share the GPU. A job
//! that allocates more than what is resident-able makes WDDM page buffers to system memory, which was
//! measured to make the GPU path slower than the CPU (UHD while another process held 10.7 of 12 GB).
//! So each job sizes its bands for min(configured share, a fraction of the currently free budget).

/// (budget, current usage) in bytes of the local (dedicated) segment for THIS process, from DXGI
/// QueryVideoMemoryInfo of the adapter with the given PCI vendor / device id. The budget shrinks when
/// other processes use the GPU. None if not available (non-Windows, no DXGI 1.4, no match).
#[cfg(windows)]
pub fn local_budget(vendor: u32, device: u32) -> Option<(u64, u64)> {
    use windows::core::Interface;
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, IDXGIAdapter3, IDXGIFactory1, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO,
    };
    unsafe {
        let f: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        for i in 0..16u32 {
            let a = match f.EnumAdapters1(i) {
                Ok(a) => a,
                Err(_) => break,
            };
            let d = match a.GetDesc1() {
                Ok(d) => d,
                Err(_) => continue,
            };
            if d.VendorId != vendor || d.DeviceId != device {
                continue;
            }
            let a3: IDXGIAdapter3 = a.cast().ok()?;
            let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            a3.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info).ok()?;
            return Some((info.Budget, info.CurrentUsage));
        }
        None
    }
}

#[cfg(not(windows))]
pub fn local_budget(_vendor: u32, _device: u32) -> Option<(u64, u64)> {
    None
}
