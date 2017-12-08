FROM ubuntu

RUN \
 apt-get update && \
 apt-get install python3.5 -y && \
 apt-get install -y python-pip python-dev build-essential && \
 pip install --upgrade pip && \
 pip install tensorflow && \
 pip install Flask && \
 apt-get install git -y && \
 pip install numpy && \
 pip install pandas && \
 git clone git clone https://github.com/prachigondhali/linear-classifier.git

RUN cd /linear-classifier

EXPOSE 9000

WORKDIR "/linear--classifier"

RUN chmod 777 /linear-classifier

CMD ["python", "linear_classifie_Pratikshar"]
