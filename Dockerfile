FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies
RUN yum update -y && yum install -y gcc g++

# Copy requirements and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY models/ ${LAMBDA_TASK_ROOT}/models/
COPY data/ ${LAMBDA_TASK_ROOT}/data/

# Copy Lambda handler
COPY src/lambda_handler.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler 
CMD [ "lambda_handler.handler" ] 