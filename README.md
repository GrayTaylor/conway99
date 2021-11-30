# conway99
Assorted python code for working with strongly regular graphs, specfically for investigating subgraphs of Conway's 99-graph, an srg(99,14,1,2) - if such a thing exists! 

The useful result here is a classification of the possible subgraphs consisting of [two adjacent vertices and their combined neighbourhoods](/Two adjacent vertices and their combined neighbourhoods.ipynb); details on how that works (and what some of the other notebooks attempt to do) are in [this blog post](https://maths.straylight.co.uk/archives/1330). Earlier posts describe [the Conway 99-graph problem](https://maths.straylight.co.uk/archives/1299) and some constraints such a graph [would have to satisfy](https://maths.straylight.co.uk/archives/1315).

For isomorphism testing I made use of Nauty via [OApackage](https://github.com/eendebakpt/oapackage):

<ul>
<li><p><a class="reference external" href="https://doi.org/10.21105/joss.01097">OApackage: A Python package for generation and analysis of orthogonal arrays, optimal designs and conference designs</a>, P.T. Eendebak, A.R. Vazquez, Journal of Open Source Software, 2019</p></li>
<li><p><em>Complete Enumeration of Pure-Level and Mixed-Level Orthogonal Arrays</em>, E.D. Schoen, P.T. Eendebak, M.V.M. Nguyen, Volume 18, Issue 2, pages 123-140, 2010.</p></li>
</ul>

