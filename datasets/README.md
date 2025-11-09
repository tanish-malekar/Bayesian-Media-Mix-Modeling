# Robyn simulated weekly data

`robyn_weekly.csv` is the 208-row simulated weekly example distributed by
[Meta Robyn](https://github.com/facebookexperimental/Robyn). The exact upstream URL,
commit, and SHA-256 are in `provenance.json`. `ROByn-LICENSE` preserves the source
MIT license and copyright notice.

The pipeline uses `DATE`, `revenue`, `tv_S`, `ooh_S`, `print_S`, `facebook_S`,
`search_S`, `competitor_sales_B`, `newsletter`, and `events`. The impression and
click columns remain available as original source fields. Spend is nonnegative,
revenue is positive, and dates form a contiguous weekly sequence.
