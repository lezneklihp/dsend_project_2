FROM python:3.6

WORKDIR /Users/phillip.kenzel/Desktop/prog_work/udacity/dsend_2

COPY requirements.txt ./
COPY --chown=root:root setup.sh /setup.sh
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 3001
RUN chmod +x /setup.sh

ENTRYPOINT ["/setup.sh"]
