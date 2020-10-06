#!/bin/sh

EXTERNAL_ALL_PACKAGES=$(cat <<EOF
    all:
      providers:
        mpi:
          - mvapich2@2.3 arch=linux-rhel7-ivybridge
        lapack:
          - openblas threads=openmp
        blas:
          - openblas threasd=openmp
      buildable: true
      version: []
EOF
)

EXTERNAL_PACKAGES=$(cat <<EOF
    cmake::
      buildable: True
      variants: ~openssl ~ncurses
      version:
      - 3.18.0
      externals:
      - spec: cmake@3.18.0 arch=linux-rhel7-ivybridge
        modules:
        -  cmake/3.18.0
    cuda::
      buildable: False
      version:
      - 10.2.89
      externals:
      - spec: cuda@10.2.89 arch=linux-rhel7-ivybridge
        modules:
        - cuda/10.2.89
    cudnn::
      buildable: true
      version:
      - 8.0.0.180-10.2-linux-x64
    gcc::
      buildable: False
      version:
      - 7.3.0
      externals:
      - spec: gcc@7.3.0 arch=linux-rhel7-ivybridge
        modules:
        - gcc/7.3.0
    hwloc::
      buildable: False
      version:
      - 2.0.2
      externals:
      - spec: hwloc@2.0.2 arch=linux-rhel7-ivybridge
        prefix: /usr/lib64/libhwloc.so
    mvapich2::
      buildable: True
      version:
      - 2.3
      externals:
      - spec: mvapich2@2.3%gcc@7.3.0 arch=linux-rhel7-ivybridge
        prefix: /usr/tce/packages/mvapich2/mvapich2-2.3-gcc-7.3.0/
    openblas::
      buildable: True
      variants: threads=openmp
      version:
      - 0.3.10
    opencv::
      buildable: true
      variants: build_type=RelWithDebInfo ~calib3d+core~cuda~dnn~eigen+fast-math~features2d~flann~gtk+highgui+imgproc~ipp~ipp_iw~jasper~java+jpeg~lapack~ml~opencl~opencl_svm~openclamdblas~openclamdfft~openmp+png~powerpc~pthreads_pf~python~qt+shared~stitching~superres+tiff~ts~video~videoio~videostab~vsx~vtk+zlib
      version:
      - 4.1.0
    python::
      buildable: True
      variants: +shared ~readline ~zlib ~bz2 ~lzma ~pyexpat
      version:
      - 3.7.2
      externals:
      - spec: python@3.7.2 arch=linux-rhel7-ivybridge
        modules:
        - python/3.7.2
    rdma-core::
      buildable: False
      version:
      - 20
      externals:
      - spec: rdma-core@20 arch=linux-rhel7-ivybridge
        prefix: /usr
EOF
)
