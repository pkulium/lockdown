## PP-06.02: String-frequency

**Adapt your previous program from PP-05.03: Duplicates, to instead print out a
horizontal [Histogram](https://en.wikipedia.org/wiki/Histogram) of the frequency distribution of A-Z letters.**
Stack the `'#'` symbol for each percentage point for a letter (e.g. if `'A'` appears 5% of the time the histogram should display `"#####"` for that bucket), and
after each stack of `'#'` print the percentage for that letter.

You should round the percentages in the histogram display, so a distribution of 5.21% should look the same as 5% in terms of `'#'` symbols.
Your program should work on any input string of length 1500 or less, with output that should match the following examples, plus the defined test cases. Note that as before, the matching should be case-insensitive, hence why the `'T'` matches with `'t'`: in the example below:

```console
Enter in a string to calculate a histogram: This is such an exciting programming problem!
Analyzing string "This is such an exciting programming problem!"

  :::Histogram:::

  a: ##### (5.26)
  b: ### (2.63)
  c: ##### (5.26)
  d:
  e: ##### (5.26)
  f:
  g: ######## (7.89)
  h: ##### (5.26)
  i: ############# (13.16)
  j:
  k:
  l: ### (2.63)
  m: ######## (7.89)
  n: ######## (7.89)
  o: ##### (5.26)
  p: ##### (5.26)
  q:
  r: ######## (7.89)
  s: ######## (7.89)
  t: ##### (5.26)
  u: ### (2.63)
  v:
  w:
  x: ### (2.63)
  y:
  z:
```

<hr/>

Some tips as you get started on this program:
* Previously, we counted the occurrences of `'A'`-`'Z'` and `'a'`-`'z'`. You can do something
very similar here, but you will also need to keep track of the total letters found so that you can
calculate the percentages as well.
* Using functions `isalpha()` and `tolower()` in <[string.h](https://en.cppreference.com/w/c/string/byte)> will make your code a bit cleaner / shorter.
* Don't overthink the rounding portion. You can use the `roundf()` or
  related functions in <[math.h](https://en.cppreference.com/w/c/numeric/math/round)>
  but this isn't necessary. By adding 0.5 to a floating-point value and then casting from `float` to
  `int`, you will get a similar result without requiring a function call.
* Printing special character `"` in `printf()` requires a backspace first: printf(`"\""`) will print a single `"` symbol. To print `"Hello there"` with the quotes, you would call `printf(""\Hello there\"");`
* The `fgets()` call includes the final newline character `'\n'`. Make sure not to print this when printing out the input string.   
