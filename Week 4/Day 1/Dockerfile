FROM python:3.9
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user ./requirements.txt $HOME/app/requirements.txt
RUN pip install -r $HOME/app/requirements.txt

COPY --chown=user . $HOME/app
RUN chown user -R ${HOME}/app/data

CMD ["chainlit", "run", "app.py", "--port", "7860"]