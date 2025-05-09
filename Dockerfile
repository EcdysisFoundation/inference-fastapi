FROM docker.io/python:3.12-slim as python

# build stage
FROM python as python-build-stage

COPY ./requirements.txt /code/requirements.txt

#RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip wheel --wheel-dir /usr/src/app/wheels -r /code/requirements.txt

# run stage
FROM python AS python-run-stage

WORKDIR /code

COPY --from=python-build-stage /usr/src/app/wheels  /wheels/

RUN pip install --no-cache-dir --no-index --find-links=/wheels/ /wheels/* \
  && rm -rf /wheels/
