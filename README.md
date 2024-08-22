# Exercício Árvores de decisão e Ensembles

## O quê contém esse trabalho?

- **Relatório sobre processo de treinamento dos modelos**: localizado em docs/relatorio_final.pdf
- **Código python relativo ao processo de treino dos modelos**: localizado em ./build_models.py
- **Servidor web**: ./main.py
- **Dockerfile**: ./Dockerfile

## Como instalar e rodar o servidor web Flask?

- baixar esse projeto em sua máquina local ou utilizar um ambiente virtual como Codespaces
- rodar o comando `sudo docker build . -t my-digit-predictor`
- quando o comando acima rodar por completo, criando uma imagem docker com sucesso, rodar o comando `sudo docker run -p 8000:8000 my-digit-predictor`

## Como utilizar o servidor web Flask para fazer inferência de dígitos?
Após instalar e rodar o servidor com sucesso, conforme os passos anteriores, fazer:

- ir para o browser local de sua máquina e entrar no endereço http://localhost:8000/
- você verá uma página web que pedirá para escolher um modelo de predição a ser utilizado e desenhar um digito na tela
- após isso, clicar no botão **Que digito é esse?**
- o usuário verá um pop-up com o resultado da inferência realizada
- o usuário também pode fazer upload de uma foto de um digito, caso nao queira desenhar um digito na tela
