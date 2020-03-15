FROM python:3.6

WORKDIR ./dsend_project_2

COPY requirements.txt ./
COPY --chown=root:root setup.sh /setup.sh
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 3001
RUN chmod +x /setup.sh

ENTRYPOINT ["/setup.sh"]
