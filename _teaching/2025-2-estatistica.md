---
title: "Estatística Fundamental"
collection: teaching
type: "Turma 2025-2"
permalink: /teaching/2025-2-estatistica
venue: "UEG, Agronomia"
date: 2025-08-13
location: "Posse, Goiás"
katex: true
---

## Ementa
Introdução à estatística. Descrição e exploração de dados. População e amostra. Distribuição de frequência. Medidas de tendência central e medidas de dispersão. 
Noções de probabilidade; Distribuição binomial, de Poisson e normal (Gauss); Teste de hipótese.


## Bibliografia Básica

1. BUSSAB, W. O.; MORETTIN, P. A. Estatística básica. 8. ed. São Paulo: Saraiva, 2013. 548p. [Link Gnuteca](https://app.minhabiblioteca.com.br/reader/books/9788571441484/)
2. CENTENO, A. J. Curso de estatística aplicada à biologia. 2. ed. Goiânia: UFG, 2002. 234 p.
3. CRESPO, A. A. Estatística fácil. 19. ed. São Paulo: Saraiva, 2009. 232p. [Link Gnuteca](https://app.minhabiblioteca.com.br/reader/books/9788571440814/)

## Referências Bibliográficas Complementares
  
4. BANZATTO, D. A.; KRONKA, S. N. Experimentação Agrícola. 4. ed. Jaboticabal: FUNEP, 2006. 237 p.
5. CALLEGARI-JACQUES, S. M. Bioestatística: princípios e aplicações. Porto Alegre: Artmed, 2007. 255p.
6. GOMES, P. F. Curso de estatística experimental.15. ed. Piracicaba: FEALQ, 2009. 451p.
7. OLIVEIRA, F. E. M. de. Estatística e Probabilidade. 3. ed. Rio de Janeiro: LTC, 2017. 280p. [Link Gnuteca](https://app.minhabiblioteca.com.br/reader/books/9788521633846/)
8. SPIEGEL, M. R; STEPHENS, L. J. Estatística. 4. ed. Porto Alegre: Artmed, 2009. 600p.


### O texto base do curso será:
 
   > PEREIRA, M. PEREIRA, P. **Notas de Aula de Estatística Aplicada à Engenharia** , 1a ed., 2020.

que está disponível para [Download Aqui](https://antmelo.github.io/files/estatistica.pdf){: .btn} 


## Listas de Exercícios para fazer e contabilizar pontos complementares

|   Página    | Seção  | Questões  | Itens | Adicionado em   |              Descrição                                  |
| --------    | -----  | -----     | ---------   | -------------   | ------------------------------------------------------- |
|     37     |  4.3  |  1, 2, 3, 4, 7, 8, 10, 11, 13, 14 | todos sem separatrizes  | 22/08   | Fazer e preparar para apresentação em sala na semana seguinte. [Respostas aqui](https://antmelo.github.io/files/ZU-EstL1.pdf){: .btn}   |
|     48     |  5.4  |  todas exceto a 24 | todos  | 02/09   | Fazer e preparar para apresentação em sala na semana seguinte (em especial aos que não foram no quadro ainda).   |


## Addendum sobre o uso do Teorema de Bayes

Para evitar confusão no uso do Teorema de Bayes, uma versão equivalente dele, porém mais simples pode ser conveniente que é o chamado *Teorema da Probabilidade Total*:

**Teorema (da Probabilidade Total):** *Suponha que $B_1, B_2, \ldots, B_n$ são eventos mutuamente exclusivos cuja união é o espaço amostral $\Omega$, ou seja, os $B_i$ formam uma **partição** de $\Omega$. Então, se $A$ é um evento qualquer, temos que:* 

$$P(A) = P(B_1 \cap A) + P(B_2 \cap A) + \ldots + P(B_n \cap A).$$

Este resultado é  utilizado quando $P(A)$ é difícil de ser calculada diretamente, porém simples se for usada a relação acima.

**Exemplo 1 (Exercício 5.4.6):** Em uma indústria de enlatados, as linhas de Produção I, II, III respondem por 50%, 30%, 20% da produção, respectivamente. As proporções de latas com defeito de produção nas linhas I, II, e III são 0, 4%, 0, 6% e 1, 2%. Qual a probabilidade de uma lata defeituosa (descoberta ao final da inspeção do produto acabado) provir da linha I?

*Solução:* Note que as linhas formam uma partição da produção com probabilides $P(I) = 0,5$, $P(II) = 0,3$ e $P(III) = 0,2$ de escolher uma lata vinda da respectiva linha. Seja $D$ o evento "escolher uma lata defeituosa". Queremos calcular $P(I \mid D)$. Pela Regra da Multiplicação, sabemos que $P(I \cap D) = 0,5 \times 0,004 = 0,002$. Usando o Teorema da Probabilidade Total, temos que 

$$
\begin{aligned}
   P(D) &= P(I \cap D) + P(II \cap D) + P(III \cap D) \\
  &= 0,5 \times 0,004 + 0,3 \times 0,006 + 0,2 \times 0,012   \\
  &= 0,002 + 0,0018 + 0,0024   \\
  &= 0,0062    \\
\end{aligned}
$$

Assim, $P(I \mid D) = \frac{P(I \cap D)}{P(D)} = \frac{0,002}{0,0062} = 0,322580645 \approx 32,26\%$.

**Exemplo 2 (Exercício 5.4.10):** Uma empresa de sementes fiscalizadas vende pacotes com 20 Kg cada.
As máquinas $A$, $B$, e $C$ enchem 25%, 35% e 40% do total produzido, respectivamente. Da produção de cada máquina 5%, 4% e 2% respectivamente, são pacotes fora do peso aceitável. Escolhe-se ao acaso um pacote e verifica-se que está fora do peso aceitável.
Qual a probabilidade de que o pacote tenha vindo da máquina $A$?

*Solução:* Note que as máquinas formam uma partição da produção com probabilides $P(A) = 0,25$, $P(B) = 0,35$ e $P(C) = 0,4$ de escolher um pacote vindo da respectiva máquina. Seja $F$ o evento "escolher um pacote fora do peso aceitável". Queremos calcular $P(A \mid F)$. Pela Regra da Multiplicação, sabemos que $P(A \cap F) = 0,25 \times 0,05 = 0,0125$. Usando o Teorema da Probabilidade Total, temos que 

$$
\begin{aligned}
   P(F) &= P(A \cap F) + P(B \cap F) + P(C \cap F) \\
  &= 0,25 \times 0,05 + 0,35 \times 0,04 + 0,4 \times 0,02   \\
  &= 0,0125 + 0,014 + 0,008   \\
  &= 0,0345    \\
\end{aligned}
$$

Assim, $P(A \mid F) = \frac{P(A \cap F)}{P(F)} = \frac{0,0125}{0,0345} = 0,362318841 \approx 36,23\%$.

Como podemos ver, é mais fácil usar o Teorema da Probabilidade Total com a Regra da Multiplicação do que montar um somatório de probabilidades condicionais no Teorema de Bayes em casos como os exemplificados.


   Dúvidas, comentários ou observações podem ser feitas por email &#129488; &#129303;.

