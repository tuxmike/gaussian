#Simple 2D - Gaussian Mixture Model

Simple **Gaussian Mixture Model** with **_Online_ Expectation Maximisation** algorithm, based on Allou Samé and Christophe Ambroise and Gérard Govaert
("An online classification EM algorithm based on the mixture model", Statistics and Computing 2007)

Usage:
```C++
GaussianMixture<2> gm;
gm.addDatapoint( x, y );
...
gm.addDatapoint( x, y );
BivGaussParams g0 = gm.getGaussian(0);
BivGaussParams g1 = gm.getGaussian(1);
gm.addDatapoint( x, y );
...
gm.addDatapoint( x, y );
BivGaussParams g1 = gm.getGaussian(1);
...
```

Copyright (c) 2016 Michael Reithmeier

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



