FROM nvidia/cuda:11.2.2-devel-ubuntu20.04 as builder

RUN chmod 0777 /tmp
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev \
        python3-pip \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

ENV CMAKE_VERSION=3.19.7
RUN wget -q https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-Linux-x86_64.tar.gz && \
    tar xf *.tar.gz && \
    rm *.tar.gz
ENV PATH=$PATH:/root/cmake-$CMAKE_VERSION-Linux-x86_64/bin

ENV ONEAPI_VERSION=2021.3.0
ENV MKL_BUILD=520
ENV DNNL_BUILD=344

RUN wget -q https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    apt-key add *.PUB && \
    rm *.PUB && \
    echo "deb https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        intel-oneapi-mkl-devel=$ONEAPI_VERSION-$MKL_BUILD \
        intel-oneapi-dnnl-devel=$ONEAPI_VERSION-$DNNL_BUILD \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY third_party third_party
COPY cli cli
COPY include include
COPY src src
COPY CMakeLists.txt .

ARG CXX_FLAGS
ENV CXX_FLAGS=${CXX_FLAGS}
ARG CUDA_NVCC_FLAGS
ENV CUDA_NVCC_FLAGS=${CUDA_NVCC_FLAGS:-"-Xfatbin=-compress-all"}
ARG CUDA_ARCH_LIST
ENV CUDA_ARCH_LIST=${CUDA_ARCH_LIST:-"Common"}
ENV CTRANSLATE2_ROOT=/opt/ctranslate2
# ENV CT2_VERBOSE=3


RUN mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=${CTRANSLATE2_ROOT} \
          -DWITH_CUDA=ON -DWITH_MKL=ON -DWITH_DNNL=ON -DOPENMP_RUNTIME=INTEL \
          -DENABLE_PROFILING=ON -ENABLE_PROFILING=ON \
          -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
          -DCUDA_NVCC_FLAGS="${CUDA_NVCC_FLAGS}" -DCUDA_ARCH_LIST="${CUDA_ARCH_LIST}" .. && \
    VERBOSE=1 make -j$(nproc) install

RUN cp -P /opt/intel/oneapi/dnnl/latest/cpu_iomp/lib/*.so* $CTRANSLATE2_ROOT/lib && \
    cp -P /opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so $CTRANSLATE2_ROOT/lib

ENV LANG=en_US.UTF-8
COPY README.md .
COPY python python

RUN cd python && \
    python3 -m pip --no-cache-dir install -r install_requirements.txt && \
    python3 setup.py bdist_wheel --dist-dir $CTRANSLATE2_ROOT

FROM nvidia/cuda:11.2.2-base-ubuntu20.04 as ct2_python

RUN chmod 0777 /tmp
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcublas-11-2 \
        python3-pip \
        vim \
        htop \
        && \
    apt-get purge -y cuda-compat-11-2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install OpenNMT-py flask_cors

ENV CTRANSLATE2_ROOT=/opt/ctranslate2
# ENV CT2_VERBOSE=3
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CTRANSLATE2_ROOT/lib

COPY --from=builder $CTRANSLATE2_ROOT $CTRANSLATE2_ROOT
RUN python3 -m pip --no-cache-dir install $CTRANSLATE2_ROOT/*.whl && \
    rm $CTRANSLATE2_ROOT/*.whl


FROM ct2_python

# COPY serv_OpenNMT-py /serv_OpenNMT-py
RUN mkdir /serv_OpenNMT-py
WORKDIR /serv_OpenNMT-py
COPY serv_OpenNMT-py/requirements.docker.txt .
RUN pip install -r requirements.docker.txt
# RUN ct2-opennmt-py-converter --model_path release_data/model-ref-warmup-8000-lr-1.7-fast-gelu_step_170000.pt --output_dir release_data/ctranslate_model --force

ENV PYTHONPATH "${PYTHONPATH}:/serv_OpenNMT"
# ENTRYPOINT ["ct2-opennmt-py-converter",  "--model_path", "/models/model-lm-medium-finetune-ubiqus-mode2-2500000-valid-lr-7_step_40000.pt", "--output_dir", "ctranslate2"]
ENV CT2_VERBOSE=3
EXPOSE 80
CMD ["python3", "./server.py", "--config", "./available_models/ctranslate_conf.json", "--port", "80"]
