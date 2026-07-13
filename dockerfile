FROM carbonsilicon-ai-registry.cn-hangzhou.cr.aliyuncs.com/teamcity/outer_save:marker_with_mol
ENV PATH /opt/conda/bin:$PATH
# 激活新环境
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

COPY requirements-otel.txt /tmp/requirements-otel.txt
RUN pip install -r /tmp/requirements-otel.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

WORKDIR /app

COPY ./ /app/
