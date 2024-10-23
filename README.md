# bias-investigation

Este repositório contém os códigos Python utilizados no projeto de investigação de viés em metadados para a classificação de câncer de pele, parcialmente desenvolvido por Gabriel Schettino Lucas como parte do Trabalho de Conclusão de Curso (TCC) em Engenharia Elétrica na Universidade Federal do Espírito Santo.

## Objetivo

O objetivo desse trabalho é investigar a presença de vieses em uma rede neural multimodal que classifica câncer de pele, focando em identificar fatores que possam comprometer a equidade das predições. Para isso, utilizamos o dataset PAD-UFES-20, que contém imagens de lesões de pele e metadados associados, como idade, sexo e localização da lesão.

## Estrutura do Repositório

- `data/`: Scripts relacionados ao pré-processamento dos dados e manipulação dos metadados do dataset PAD-UFES-20.
- `models/`: Implementação dos modelos de Redes Neurais Convolucionais (CNNs) utilizados para a classificação de câncer de pele.
- `experiments/`: Scripts e notebooks relacionados aos experimentos realizados para a investigação dos vieses.
- `analysis/`: Scripts para análise estatística e visualização dos resultados, incluindo gráficos que mostram o impacto dos metadados na predição.
- `notebooks/`: Notebooks Jupyter que documentam as etapas de análise, validação e resultados intermediários do projeto.
- `utils/`: Funções auxiliares para visualização, carregamento de dados e métricas de avaliação.
- `results/`: Armazenamento dos resultados das análises, incluindo gráficos, tabelas e outros outputs gerados.

## Execução

- Faça o setup do [gandalf](https://github.com/life-ufes/gandalf) e do [raug](https://github.com/paaatcha/raug) na sua máquina, utilizando Linux ou o WSL. As instruções estão disponíveis no repositório do projeto.
- No arquivo `config.json`, localizado na pasta `utils`, configure os caminhos até os arquivos do PAD-UFES-20, do MetaBlock, Raug e do checkpoint do modelo treinado.