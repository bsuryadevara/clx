# Copyright (c) 2020, NVIDIA CORPORATION.

# An integration test & dev container which builds and installs CLX from default branch
FROM rapidsai/rapidsai-dev:0.17-cuda10.1-devel-ubuntu16.04-py3.7

# Add everything from the local build context
ADD . /rapids/clx/
RUN chmod -R ugo+w /rapids/clx

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
ENV KAFKA_VERSION 2.5.1
ENV KAFKA_HOME /opt/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION"
ENV CLX_STREAMZ_HOME /opt/clx_streamz

ADD examples/streamz/scripts "$CLX_STREAMZ_HOME"/scripts
ADD examples/streamz/python "$CLX_STREAMZ_HOME"/python
ADD examples/streamz/resources "$CLX_STREAMZ_HOME"/resources

RUN mkdir -p "$CLX_STREAMZ_HOME"/ml/models/cybert && \
	mkdir "$CLX_STREAMZ_HOME"/ml/models/dga && \
	mkdir "$CLX_STREAMZ_HOME"/data

RUN wget -q https://downloads.apache.org/kafka/2.5.1/kafka_2.13-2.5.1.tgz -O /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz && \
        tar xfz /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz -C /opt && \
        rm /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz

# Download cybert apache model from huggingface for example
RUN wget -q http://models.huggingface.co.s3.amazonaws.com/bert/raykallen/cybert_apache_parser/config.json -O "$CLX_STREAMZ_HOME"/ml/models/cybert/config.json
RUN wget -q http://models.huggingface.co.s3.amazonaws.com/bert/raykallen/cybert_apache_parser/pytorch_model.bin -O "$CLX_STREAMZ_HOME"/ml/models/cybert/pytorch_model.bin

# Download apache logs
RUN wget -q https://rapidsai-data.s3.us-east-2.amazonaws.com/cyber/clx/apache_raw_sample_1k.txt -O "$CLX_STREAMZ_HOME"/data/apache_raw_sample_1k.txt

# Download dga detection model and sample input data
RUN wget -q https://rapidsai-data.s3.us-east-2.amazonaws.com/cyber/clx/dga_detection_pytorch_model.bin -O "$CLX_STREAMZ_HOME"/ml/models/dga/pytorch_model.bin
RUN wget -q https://rapidsai-data.s3.us-east-2.amazonaws.com/cyber/clx/dga_detection_input.jsonlines -O "$CLX_STREAMZ_HOME"/data/dga_detection_input.jsonlines

# Zookeeper
EXPOSE 2181

# Kafka
EXPOSE 9092

RUN source activate rapids && \
    conda install -c pytorch cudatoolkit=10.1 "pytorch==1.7.0" torchvision "cudf_kafka=0.17" "custreamz=0.17" "scikit-learn>=0.21" ipywidgets python-confluent-kafka transformers "seqeval=0.0.12" python-whois seaborn requests matplotlib pytest jupyterlab "openjdk=8.0.152" dask-cuda && \
    pip install "git+https://github.com/rapidsai/cudatashader.git" && \
    pip install mockito && \
    pip install wget && \
    pip install elasticsearch && \
    pip install elasticsearch-async

RUN source activate rapids \
  && conda install -n rapids jupyterlab-nvdashboard \
  && jupyter labextension install @jupyter-widgets/jupyterlab-manager dask-labextension jupyterlab-nvdashboard

# clx build/install
RUN source activate rapids && \
    cd /rapids/clx/python && \
    python setup.py install

# clx_streamz_tools install
RUN source activate rapids && \
	cd "$CLX_STREAMZ_HOME"/python && \
	python setup.py install
	
WORKDIR /rapids/clx

RUN groupadd gtxgroup -g 7207 && useradd gtxadmin -u 7206 -g gtxgroup -d /opt/nvidia/gtxadmin
RUN mkdir -p /opt/nvidia
RUN chown -R gtxadmin:gtxgroup /opt/nvidia && \
    chown -R gtxadmin:gtxgroup $CLX_STREAMZ_HOME && \
    chown -R gtxadmin:gtxgroup $KAFKA_HOME && \
    chown -R gtxadmin:gtxgroup /rapids
    
VOLUME /opt/nvidia

USER gtxadmin
