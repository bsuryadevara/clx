# Copyright (c) 2020, NVIDIA CORPORATION.

# An integration test & dev container which builds and installs CLX from default branch
FROM rapidsai/rapidsai:cuda10.1-runtime-ubuntu18.04-py3.8

RUN mkdir -p /opt/nvidia/gtxadmin && \
    groupadd gtxgroup -g 7207 && \
    useradd gtxadmin -u 7206 -g gtxgroup -d /opt/nvidia/gtxadmin

# Add everything from the local build context
ADD . /opt/nvidia/gtxadmin/clx/

RUN chmod -R ugo+w /opt/nvidia/gtxadmin/clx

RUN apt update -y --fix-missing && \
    apt upgrade -y

RUN apt-get install -y librdkafka-dev \
        krb5-user \
        vim \
        wget \
        dnsutils \
        net-tools \
        gdb \
        build-essential \
        valgrind \
        unzip && \
    apt-get clean

ENV SCALA_VERSION 2.13
ENV KAFKA_VERSION 2.7.0
ENV CLX_HOME /opt/nvidia/gtxadmin/clx
ENV CLX_STREAMZ_HOME /opt/nvidia/gtxadmin/clx_streamz
ENV KAFKA_HOME /opt/nvidia/gtxadmin/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION"

ADD examples/streamz/python "$CLX_STREAMZ_HOME"/python
ADD examples/streamz/scripts "$CLX_STREAMZ_HOME"/scripts
ADD examples/streamz/resources "$CLX_STREAMZ_HOME"/resources

RUN mkdir -p "$CLX_STREAMZ_HOME"/ml/models/cybert && \
	mkdir "$CLX_STREAMZ_HOME"/ml/models/dga && \
    mkdir "$CLX_STREAMZ_HOME"/ml/models/bert-base-cased

RUN wget -q https://downloads.apache.org/kafka/2.7.0/kafka_2.13-2.7.0.tgz -O /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz && \
        tar xfz /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz -C /opt/nvidia/gtxadmin && \
        rm /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz

RUN wget -q https://huggingface.co/bert-base-cased/resolve/main/config.json -O "$CLX_STREAMZ_HOME"/ml/models/bert-base-cased/config.json
RUN wget -q https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin -O "$CLX_STREAMZ_HOME"/ml/models/bert-base-cased/pytorch_model.bin

# Zookeeper
EXPOSE 2181

# Kafka
EXPOSE 9092

RUN source activate rapids && \
    conda install -c pytorch cudatoolkit=10.1 "pytorch==1.7" torchvision "cudf_kafka=0.17" "custreamz=0.17" "scikit-learn>=0.21" "nodejs>=12" ipywidgets python-confluent-kafka "transformers=3.5.*" "seqeval=0.0.12" python-whois seaborn requests matplotlib pytest jupyterlab "openjdk=8.0.152" dask-cuda && \
    pip install "git+https://github.com/rapidsai/cudatashader.git" && \
    pip install mockito && \
    pip install wget && \
    pip install elasticsearch && \
    pip install elasticsearch-async

RUN source activate rapids && \
    conda install -n rapids jupyterlab-nvdashboard && \ 
    jupyter labextension install @jupyter-widgets/jupyterlab-manager dask-labextension jupyterlab-nvdashboard

# clx build/install
RUN source activate rapids && \
    cd "$CLX_HOME"/python && \
    python setup.py install

# clx_streamz_tools install
RUN source activate rapids && \
	cd "$CLX_STREAMZ_HOME"/python && \
	python setup.py install


VOLUME /opt/nvidia
USER gtxadmin

WORKDIR ${CLX_STREAMZ_HOME}

CMD ["/bin/bash"]