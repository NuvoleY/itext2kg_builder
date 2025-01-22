FROM python:3.10

WORKDIR /itext2kg_builder

ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "build.py"]