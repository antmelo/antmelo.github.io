---
permalink: /
layout: post
title: "Saudações!"
katex: True
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Se você é estudante de uma das minhas turmas, não deixe de conferir a seção [Disciplinas](https://antmelo.github.io/teaching/){: .btn .btn--success} para eventuais atualizações de conteúdo. Se você já é mais experiente e quer se entreter ou apenas passar o tempo, talvez a seção [Publicações](https://antmelo.github.io/publications/){: .btn .btn--success} seja interessante.

## Independente do seu nível, minha primeira dica é: aprenda $$LaTeX$$ !

$$\LaTeX$$ é uma implementação da linguagem TeX, criada para perfeita simbologia matemática, e por isso, é um sistema de preparação de documentos amplamente utilizado por pessoas na área das ciências e engenharias. Diferentemente de editores como Word, $$LaTeX$$ não apresenta uma interface amigável onde se vê a edição em tempo real. Mas há vantagens em fazer isso:
```R
$$
\begin{aligned}
  & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
  = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
  & (x_1, \ldots, x_n) \left( \begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)
  \left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
\end{aligned}
$$
```
virar isso

$$
\begin{aligned}
  & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
  = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
  & (x_1, \ldots, x_n) \left( \begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)
  \left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
\end{aligned}
$$


Já passou pelo drama de dar “enter” num documento Word e uma imagem saltar duas páginas, não já? Ou ter de organizar referências bibliográficas uma por uma? Por essa razão, muitas pessoas se perguntam:

> Por que eu deveria abandonar algo simples e adotar algo mais complexo?

Abaixo vão 12 razões — algumas retiradas e adaptadas deste [site](http://web.mit.edu/klund/www/urk/texvword.html).

1. O padrão matemático em TeX gera equações e funções corretamente formatadas. Em Word, o editor de equações está longe de ser ideal.

2. TeX não apresenta bugs — o Word, como sabemos, está recheado de bugs.

3. TeX é gratuito e livre, você não terá qualquer custo com isso.

4. Em TeX, você pode comentar o seu código/texto no mesmo espaço em que seu conteúdo é gerado, o que é bem útil em programação.

5. TeX oferece uma linguagem completa. Ou seja: você pode criar funções que efetuam um procedimento para você (muitas dessas funções não podem ser criadas via macros em Word).

6. Não há vírus de macros em TeX. Ou seja: maior segurança.

7. Não há incompatibilidade de versões: se você criou um arquivo TeX em 1995, conseguirá abri-lo perfeitamente hoje.

8. LaTeX oferece uma maneira independente de lidar com bibliografias. Nada de comprar EndNote ou algo parecido: toda a sua biblioteca de referências é mantida em um simples arquivo, ao qual você conecta citações.
> **Bônus:** você pode facilmente reunir suas referências com [Zotero](https://www.zotero.org/) e apenas exportá-las em BibTeX para seu trabalho.

9. Documentos em TeX são pequenos e rápidos.

10. LaTeX é o padrão científico/acadêmico em diversas áreas do conhecimento — e nos maiores centros acadêmicos do mundo.

11. LaTeX gera documentos mais aprimorados esteticamente, com menos hifenizações e menos espaçamentos exagerados entre palavras.

12. Seu pdf é gerado com uma estrutura interna, em que você acessa seções via links — isso é feito automaticamente com um pacote.

Para estudantes de graduação, há duas opções para produzir seus documentos:

+ Tornar-se um usuário sofisticado de Word (ou similar). Minimamente, conheça bem os estilos do Word, numeração automática de seções, referências entre figuras, exemplos e seções. Além disso, você precisará saber utilizar muito bem softwares como EndNote, para lidar com bibliografias e outros elementos.

+ Aprender a usar LaTeX, um sistema excelente de compilação de texto que cria documentos acadêmicos com um alto padrão de qualidade. 

E não, você não vai precisar inventar a roda do zero, pois há vários modelos prontos na internet para serem adaptados às suas necessidades. Por exemplo, na página da [abnTeX](https://www.abntex.net.br/) há uma seção de download dos modelos canônicos onde você pode encontrar modelos para [artigos](http://mirrors.ctan.org/macros/latex/contrib/abntex2/doc/examples/abntex2-modelo-artigo.pdf), [livros](http://mirrors.ctan.org/macros/latex/contrib/abntex2/doc/examples/abntex2-modelo-livro.pdf), [projetos de pesquisa](https://www.abntex.net.br/) e [trabalhos acadêmicos de conclusão de curso](https://linorg.usp.br/CTAN/macros/latex/contrib/abntex2/doc/examples/abntex2-modelo-trabalho-academico.pdf) devidamente organizados e formatados dentro das normas da ABNT.

Outra opção interessante é o [Overleaf](https://pt.overleaf.com/) que é uma plataforma de edição em $$LaTeX$$ totalmente online e que, além de ter uma seção repleta de [modelos](https://pt.overleaf.com/latex/templates) prontos para serem editados e conta gratuita, também disponibiliza um bom [tutorial](https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes) em inglês de como usar os comandos básicos e demais personalizações do sistema.
  
