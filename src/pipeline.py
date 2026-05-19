from prefect import flow, task
import subprocess

@task
def train():
    subprocess.run(['python', 'src/train.py'], check=True)

@task
def batch():
    subprocess.run(['python', 'src/batch.py'], check=True)

@task
def monitor():
    subprocess.run(['python', 'src/monitor.py'], check=True)

@flow
def churn_pipeline():
    train()
    batch()
    monitor()

if __name__ == "__main__":
    churn_pipeline() 