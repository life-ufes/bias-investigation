# bias-investigation
![gradcam](results/gradcam_gifs/MEL/PAT_995_1867_165.png.gif)
![gradcam](results/gradcam_gifs/MEL/PAT_109_868_723.png.gif)
![gradcam](results/gradcam_gifs/MEL/PAT_966_1825_546.png.gif)
![gradcam](results/gradcam_gifs/MEL/PAT_995_1867_5.png.gif)

Este repositório contém os códigos Python utilizados no projeto de investigação de viés em metadados para a classificação de câncer de pele, parcialmente desenvolvido por Gabriel Schettino Lucas como parte do Trabalho de Conclusão de Curso (TCC) em Engenharia Elétrica na Universidade Federal do Espírito Santo.

## Objetivo

O objetivo desse trabalho é investigar a presença de viés em uma rede neural multimodal que classifica câncer de pele, focando em identificar fatores que possam comprometer a equidade das predições. Para isso, utilizamos o dataset PAD-UFES-20, que contém imagens de lesões de pele e metadados associados, como idade, sexo e localização da lesão.

## Estrutura do Repositório

- `experiments/`: Scripts e notebooks relacionados aos experimentos realizados para a investigação dos vieses.
- `utils/`: Funções auxiliares para visualização, carregamento de dados e métricas de avaliação.
- `results/`: Armazenamento dos resultados das análises, incluindo gráficos, tabelas e outros outputs gerados.

## Execução

- Faça o setup do [gandalf](https://github.com/life-ufes/gandalf) e do [raug](https://github.com/paaatcha/raug) na sua máquina, utilizando Linux ou o WSL. As instruções estão disponíveis no repositório do projeto.
- No arquivo `config.json`, localizado na pasta raiz do projeto, configure os caminhos até os arquivos do PAD-UFES-20, do MetaBlock, Raug e do checkpoint do modelo treinado.