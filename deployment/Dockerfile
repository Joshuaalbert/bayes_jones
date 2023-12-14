FROM python:3.11.7


WORKDIR /code

# requirements rarely change
COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt

COPY ./requirements-tests.txt /code/requirements-tests.txt
RUN pip install -r /code/requirements-tests.txt

# will change frequently
COPY . /code
RUN pip install /code


ENTRYPOINT ["pytest", "-s", "/code"]