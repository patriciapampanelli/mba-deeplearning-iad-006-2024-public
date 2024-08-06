# Exercício para sala de aula - Árvores de decisão e Ensembles

## Introdução e objetivos
Este notebook é um material complementar à disciplina de Aprendizado de Máquinas 2. O objetivo é reforçar os conteúdos das últimas aulas e trazer alguns aspectos práticos da implementação destes modelos. Vamos exercitar algumas etapas do MLOps, incluindo a dockerização da solução, inferência e cliente, além de cobrir as etapas de treinamento, avaliação e deploy.

Neste notebook, vamos explorar como colocar em prática os conceitos aprendidos em sala de aula, desde a preparação dos dados até a implementação de modelos de aprendizado de máquina. Além disso, vamos abordar como criar uma solução escalável e pronta para produção, utilizando técnicas de dockerização e deploy.

Ao longo deste notebook, você aprenderá a:
- Preparar os dados para treinamento e avaliação de modelos
- Implementar modelos de aprendizado de máquina utilizando bibliotecas populares
- Avaliar o desempenho dos modelos e realizar ajustes necessários
- Dockerizar a solução para facilitar a implantação em diferentes ambientes
- Implementar um cliente para realizar inferências nos modelos treinados

## Preparação - Pré-aula
A execução deste notebook será feita através do Codespace do GitHub. Para isso, é necessário realizar as seguintes etapas:
- **Fork do repositório base**: Faça um fork do repositório base para criar uma cópia do projeto em sua conta do GitHub.
- **Inicialização do Codespace**: Acesse o repositório forkado e clique em "Codespaces" no botão "Code" do GitHub, em seguida, clique em "New codespace" para inicializar o ambiente de desenvolvimento. O Codespace é um ambiente de desenvolvimento hospedado na nuvem que permite que você desenvolva e execute seu código diretamente no navegador, sem a necessidade de configurar um ambiente local. Com o Codespace, você pode criar, editar e executar seu código em um ambiente isolado e seguro.
- **Build da imagem Docker:** No Codespace, localize o terminal na parte inferior da janela e copie e cole o comando docker build -t imagem-docker . nele. Em seguida, pressione Enter para executar o comando. O terminal irá exibir as etapas de construção da imagem Docker, incluindo a instalação de dependências e a compilação do código. Se o comando for executado com sucesso, você verá uma mensagem indicando que a imagem foi construída com sucesso, como por exemplo: "Successfully built <id_da_imagem>". Isso significa que a imagem Docker foi criada e está pronta para ser usada.
- **Execução do notebook:** No Codespace, localize o arquivo `./notebooks/começar_por_aqui.ipynb` e clique nele para abrir. Em seguida, clique no botão "Run" para executar o notebook. O notebook irá executar as células e exibir o resultado com a precisão do modelo. Não se preocupe. Este código é só um teste para garantir que o Python está executando corretamente no "Codespaces".

Esses passos são necessários para garantir que o mínimo do ambiente está funcional para os exercícios em sala. Se você tiver alguma dúvida ou precisar de ajuda, por favor, entre em contato comigo.

## Roteiro do exercício
#### Exercício 1 - Treinamento do modelo baseado em árvore de decisão
Neste exercício, você irá treinar um modelo de árvore de decisão utilizando o conjunto de dados MNIST. O objetivo é entender como funciona o processo de treinamento de um modelo de árvore de decisão e como ajustar os parâmetros para melhorar o desempenho do modelo.

**Você irá:**
1. Carregar o conjunto de dados MNIST e pré-processar os dados, se necessário.
2. Selecionar a biblioteca ou framework que você irá utilizar (por exemplo, Scikit-learn e/ou XGBoost).
3. Configurar os parâmetros do modelo, como a profundidade da árvore, o número de características a considerar em cada nó, etc.
4. Treinar o modelo utilizando o conjunto de dados de treinamento.

*Dicas:*
- Certifique-se de que os dados estejam pré-processados corretamente antes de treinar o modelo.
- Ajuste os parâmetros do modelo, como a profundidade da árvore, para melhorar o desempenho do modelo.

#### Exercício 2 - Avaliação dos ganhos com a utilização de modelos Ensemble
Neste exercício, você irá avaliar os ganhos obtidos ao utilizar modelos Ensemble em comparação com o modelo de árvore de decisão simples. Você irá treinar um modelo Ensemble e comparar os resultados com o modelo de árvore de decisão treinado anteriormente.

*Dicas:*
- Utilize diferentes tipos de modelos Ensemble, como Random Forest e XGBoost, para comparar os resultados.

#### Exercício 3 - Visualização da árvore de decisão e Medida de Impureza
Neste exercício, você irá explorar a estrutura da árvore de decisão e entender como a medida de impureza é utilizada para avaliar a qualidade das divisões nos nós da árvore. A medida de impureza é um conceito fundamental em árvores de decisão, pois ajuda a determinar a melhor forma de dividir os dados em subconjuntos menores e mais homogêneos.

**Você irá:**
1. Visualizar a árvore de decisão treinada no exercício anterior, utilizando ferramentas como o Scikit-learn ou o Graphviz.
2. Calcular a medida de impureza para diferentes conjuntos de dados e entender como ela impacta o desempenho do modelo.
3. Analisar como a medida de impureza afeta a escolha dos atributos para dividir os dados em cada nó da árvore.
4. Entender como a medida de impureza pode ser utilizada para evitar overfitting e melhorar a generalização do modelo.

#### Exercício 4 - Preparação do container para deploy do modelo
Neste exercício, você irá aprender a preparar um container Docker para deploy do modelo de árvore de decisão treinado. Isso envolve criar um ambiente isolado e portável que possa ser facilmente implantado em diferentes contextos.

Você irá:
1. Criar um arquivo Dockerfile que define as instruções para construir a imagem do container.
2. Especificar as dependências necessárias para o modelo, incluindo bibliotecas e frameworks.
3. Configurar o ambiente de execução do modelo, incluindo a definição de variáveis de ambiente e a configuração de portas.
4. Construir a imagem do container utilizando o comando `docker build`.
5. Executar o container utilizando o comando `docker run` e testar a imagem.

*Dicas:*
- Certifique-se de que o arquivo Dockerfile esteja configurado corretamente para construir a imagem do container.

#### Exercício 5 - Deploy do modelo usando Flask utilizando uma imagem docker
Neste exercício, você irá aprender a deployar o modelo de árvore de decisão treinado utilizando o framework Flask e uma imagem Docker. Isso envolve criar uma API REST que permita realizar inferências com o modelo e entender como funciona o processo de deploy de um modelo em um ambiente de produção.

Você irá:
1. Criar uma aplicação Flask que carregue o modelo treinado e o utilize para realizar inferências.
2. Definir uma API REST que permita enviar solicitações de inferência ao modelo e receber respostas.
3. Configurar a aplicação Flask dentro da imagem Docker do exercício anterior.
4. Deployar a aplicação Flask no container Docker e testar a API REST.
5. Entender como funciona o processo de deploy de um modelo em um ambiente de produção, incluindo a configuração de variáveis de ambiente e a gestão de dependências.

#### Exercício 6 - Notebook cliente com inferência direto do servidor
Neste exercício, você irá aprender a criar um notebook cliente que realize inferências direto do servidor onde o modelo de árvore de decisão foi deployado. Isso envolve entender como funciona o processo de comunicação entre o cliente e o servidor e como realizar inferências com o modelo deployado.

Você irá:
1. Criar um notebook cliente.
2. Importar as bibliotecas necessárias para realizar inferências com o modelo, incluindo a biblioteca `requests` para enviar solicitações HTTP ao servidor criado no exercício anterior.
3. Configurar o notebook cliente para se conectar ao servidor onde o modelo foi deployado.
4. Enviar solicitações de inferência ao servidor utilizando a API REST criada no exercício anterior.
5. Receber as respostas do servidor e visualizar os resultados das inferências.


## Árvores de Decisão
Lembrando que as árvores de decisão são modelos bastante utilizados para desenvolvmento de modelos de Machine Learning. Em especial, os métodos que utilizam as árvores com abordagens Ensembles resolvem uma gama de problemas bem grande tanto para classificação quanto para regressão.
![image](https://github.com/user-attachments/assets/5ce90cdb-c730-4157-aff4-2728df617401)

## Dataset
As árvores podem ser utilizadas tantos para regressão quanto para classificação. Neste exercício vamos verificar o uso da Árvore de Decisão para a classifição de dígitos.

Para este exercício vamos utilizar o dataset bastante conhecido, chamado MNIST. O dataset consiste em imagens de 64 pixels (8x8) de dígitos escritos a mão. O objetivo é classificar estas imagens em dez classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.

Este dataset é built-in no pacote Sklearn. Neste link vocês podem verificar outros datasets built-in: https://scikit-learn.org/stable/datasets/toy_dataset.html.

Alguns exemplos:

![image](https://github.com/user-attachments/assets/01868a0f-7cf8-464b-b299-901b78fdba63)

![image](https://github.com/user-attachments/assets/95d84022-bff5-43c6-a07a-3e1da52df06e)


