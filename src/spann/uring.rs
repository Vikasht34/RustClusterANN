/// io_uring support for Linux
#[cfg(all(target_os = "linux", feature = "tokio-uring"))]
use tokio_uring::fs::File;
#[cfg(all(target_os = "linux", feature = "tokio-uring"))]
use std::io;

#[cfg(all(target_os = "linux", feature = "tokio-uring"))]
pub struct UringReader {
    file: File,
}

#[cfg(all(target_os = "linux", feature = "tokio-uring"))]
impl UringReader {
    pub async fn open(path: &str) -> io::Result<Self> {
        let file = File::open(path).await?;
        Ok(Self { file })
    }
    
    pub async fn read_at(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        let buf = vec![0u8; len];
        let (res, buf) = self.file.read_at(buf, offset).await;
        res?;
        Ok(buf)
    }
}

#[cfg(not(all(target_os = "linux", feature = "tokio-uring")))]
pub struct UringReader;

#[cfg(not(all(target_os = "linux", feature = "tokio-uring")))]
impl UringReader {
    pub async fn open(_path: &str) -> std::io::Result<Self> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "io_uring only supported on Linux with tokio-uring feature",
        ))
    }
    
    pub async fn read_at(&self, _offset: u64, _len: usize) -> std::io::Result<Vec<u8>> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "io_uring only supported on Linux with tokio-uring feature",
        ))
    }
}
