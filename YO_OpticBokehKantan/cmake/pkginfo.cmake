# Writes Contents/PkgInfo of an After Effects plug-in bundle: package type eFKT, creator FXTC (8 bytes, no newline).
# cmake -DOUT=<bundle>/Contents/PkgInfo -P pkginfo.cmake
file(WRITE ${OUT} "eFKTFXTC")
