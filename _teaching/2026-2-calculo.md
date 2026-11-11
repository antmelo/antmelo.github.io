---
title: "Cálculo para Agronomia"
collection: teaching
type: "Turma 2026-2"
permalink: /teaching/2026-2-calculo
venue: "UEG, Agronomia"
date: 2026-08-14
location: "Posse, Goiás"
katex: true
---

## Ementa
Limites. Continuidade. Derivadas. Aplicação da Derivada. Máximos e Mínimos. Integral definida. Cálculo de área e integral indefinida.

## Bibliografia Básica


1. FLEMING, D. M.; GONÇALVES, M. B. Cálculo A: Funções, limite, derivação e integração. 6a ed. São Paulo: Editora Prentice Hall, 2006. 448p.
2. GUDORIZZI, H. L. Um Curso de Cálculo. vol. 1 , 5a ed. Rio de Janeiro: LTC, 2011. 607p.
3. SVIERCOSKI, R. F. Matemática aplicada às ciências agrárias. Viçosa: Editora UFV, 2008. 333p.

## Referências Bibliográficas Complementares

4. ÁVILA, G. Cálculo das Funções de uma Variável. v. 1, 7. ed. Rio de Janeiro: Livros Técnicos e Científicos, 2004. 244p
5. BATSSHELET. E. Introdução à Matemática para Biocientistas. São Paulo: ED. da Universidade de São Paulo,1978. 598p.
6. HOFFMANN, L. D. Cálculo: um curso Moderno e suas Aplicações. Rio de Janeiro: L.T.C, 2002. 525p.
7. LEITHOLD, L. O. Cálculo com Geometria Analítica. vol. 1 e 2. São Paulo: Harbra, 1994.
8. STEWART, J. Cálculo. vol. 1, 7a ed. São Paulo: Editora Thomson, 2013. 661p.

### O texto base do curso será:
 
   > MELO, A. **Cálculo Rápido para Ciências Agronômicas.** , 1a ed. Anápolis: Editora UEG, 2027.

que está disponível em versão preliminar para [Download Aqui](https://antmelo.github.io/files/CRCA.pdf){: .btn} 

Leitura complementar recomendada: 

  > PATRÃO, M. **Cálculo 1: derivada e integral em uma variável**. Brasília : Editora Universidade de Brasília, 2011. 319 p.
que está disponível em versão preliminar para [Download Aqui](https://repositorio.unb.br/bitstream/10482/7183/1/LIVRO_C%c3%a1lculo%201.pdf){: .btn} 

## Listas de Exercícios para fazer e contabilizar pontos complementares

| Seção  | Questões  | Itens | Adicionado em   |              Descrição                                  |
| -----  | -----     | ------ | -------------  | ------------------------------------------------------- |
| 2.1.4  |  todas | todos  | 14/08   | Fazer e treinar para revisão em sala na semana seguinte.   |
| 2.2.2  |  todas | todos  | 28/08   | Fazer e treinar para revisão em sala na semana seguinte.   |
| 2.3.3  |  todas | todos  | 05/09   | Fazer e treinar para revisão em sala no dia 18/09.   |
| 2.4.2  |  todas | todos  | 05/09   | Fazer e treinar para revisão em sala na dia 18/09.   |
| 3.1.3  |  todas | todos  | 15/09   | Fazer e treinar para revisão em sala na dia 25/09.   |
| [LISTA](https://antmelo.github.io/files/ZZA-CAL1-L3esp.pdf){: .btn}  |   |   | 19/09   |  Bônus: lista com todo o conteúdo da primeira avaliação bimestral com respostas.  |
| 3.2.2  |  todas | todos  | 12/10   | Fazer e treinar para revisão em sala na dia 23/10.   |
| 4.1.8  |  todos | todos  | 25/10   | Revisar com Máximos e Mínimos e treinar para revisão em sala na dia 6/11 (sem testinho). Por isso, foquem em trazer dúvidas, pois, pelo visto dia 23/10, são muitas.  |
| 4.1.8  |  todos | todos  | 25/10   | Revisar com Máximos e Mínimos e treinar para revisão em sala na dia 6/11 (sem testinho). Por isso, foquem em trazer dúvidas, pois, pelo visto dia 23/10, são muitas.  |
| 4.2.5  |  todos | todos  | 6/11   | Revisar com integrais indefinidas e treinar para revisão em sala na dia 13/11.  |
| [LISTA](https://antmelo.github.io/files/ZZS-CAL1-QS.pdf){: .btn}  |   |   | 11/11   |  Bônus: lista com todo o conteúdo da segunda avaliação bimestral com respostas.  |

![](https://media1.tenor.com/m/ctx-Bq2LBcIAAAAC/bar100r.gif)

## Respostas rápidas da última atividade em sala

O problema da lata já estava feito no Exemplo 3.18 (paǵina 64 do livro) com um volume conhecido, então fizemos a versão para um volume qualquer $V_0$. 

Sendo $r$ e $h$ o raio da base e a altura da lata, respectivamente, o volume da lata é dado por
$$
V_0 = \pi r^2 h 
$$
e a área lateral é
$$
A = 2 \pi r^2 + 2\pi rh 
$$

Como o volume é fixo, o consideramos como restrição e resolvendo a equação do volume para $h$ temos:
$$
h= \frac{V_0}{\pi r^2} 
$$
e substituindo na equação da área, temos
$$ 
A(r) = 2\pi r^2 + 2\frac{V_0}{r} 
$$

Claramente o raio deve ser positivo, e a área deve estar definida temos que  $0 < r < \infty$. Assim, para calcular os pontos críticos, derivamos a área:

$$ 
A'(r) =4 \pi r - 2\frac{V_0}{r^2} = \frac{ 2(2\pi r^3  - V_0)}{r^2} 
$$

e, portanto, a derivada é igual a zero em:

$$
r = \frac{V_0}{2\pi}^{\frac{1}{3}} 
$$

Como existe apenas um número crítico, podemos considerar o Teorema do Ponto Crítico para garantir que este mínimo é absoluto.

Além deste, as primitivas (integrais indefinidas) deveriam ser óbvias: 

$$\int( 7\cos (x) + 9e^x) \; dx  =  7 \sin (x) + 9e^x + C $$

$$ \int (21^z - \frac{1}{3} z^{-5})\; dz = \frac{21^z}{ \ln (21)} + \frac{1}{12} z^{-4} +C  $$


<iframe width="560" height="315" src="https://www.youtube.com/embed/l0Aqp5lJbqE?si=I_c0f5gGPSmH7qMB" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

 O real motivo de a matemática parecer impossível não tem nada a ver com sua inteligência, com seu professor ou com seu livro didático. O verdadeiro inimigo é algo chamado *armadilha do observador*, e todo aluno cai nela em cada sessão de estudo sem nunca perceber. Quando você lê um exemplo resolvido, seu cérebro observa a solução se desenrolar. Parece familiar. Parece compreendido. Então, você segue em frente. Mas, eis
o que seu cérebro não fez. Ele não aprendeu como resolver matemática. Ele aprendeu como observar a matemática sendo resolvida. É como assistir
alguém nadar por 6 horas e depois pular na parte funda esperando saber nadar. Observar não é o mesmo que saber. E as provas de matemática não
testam o que você observou, elas testam o que você consegue fazer. Existe uma diferença enorme entre ver uma solução e ser capaz de produzir
uma sob pressão. Os melhores alunos descobriram isso, acidental ou deliberadamente. Os alunos que mais sofrem nas provas de matemática não são os que menos estudaram. São os que passaram mais horas com seus livros didáticos. Aqueles que leram cada exemplo, destacaram cada fórmula,
fizeram as anotações mais detalhadas e ainda assim travaram quando a prova chegou. Aqui está a reviravolta sobre a qual ninguém na educação fala. Ler matemática e fazer matemática são duas atividades cerebrais completamente diferentes. Quando você lê um exemplo resolvido, seu cérebro observa a
solução. Quando você resolve um problema sozinho, seu cérebro constrói a solução. Observar nunca desenvolve a habilidade. Somente construir desenvolve a habilidade. Seu cérebro tem dois modos de aprendizado de matemática. *Modo passivo*: ler, observar, destacar (A informação
entra, parece familiar, mas nenhuma habilidade é desenvolvida). *Modo ativo*: tentar, esforçar-se, resolver (O cérebro constrói caminhos neurais, o
pensamento matemático real se forma). Matemática é uma habilidade motora, exatamente como dirigir ou nadar. Você não pode aprender a dirigir lendo
sobre dirigir. Você não pode aprender a nadar assistindo a vídeos de natação. E você não pode aprender matemática lendo exemplos resolvidos.
Seu cérebro só desenvolve a capacidade matemática quando é forçado a produzir respostas, não a consumi-las. Cientistas chamam o estado de aprendizado ativo de **zona de esforço**, o estado mental exato onde seu cérebro constrói habilidade real em vez de confiança falsa. Estudar de
forma confortável equivale a confiança falsa. Estudar na zona de esforço equivale a habilidade real. Cada minuto em que a matemática parece
difícil, você está na zona de esforço. E a zona de esforço é exatamente onde os melhores alunos vivem.
{: .notice} 



![](https://i.makeagif.com/media/5-21-2023/mBCetJ.gif)


   Dúvidas, comentários ou observações podem ser feitas por email &#129488; &#129303;.





