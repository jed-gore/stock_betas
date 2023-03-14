<<<<<<< HEAD
# serlialized_python_objects
# webcrawler_class
=======
## SQ starting to trade less like a bitcoin stock and more like a processor 

I recall a couple years ago I was looking at SQ and how much the stock traded with bitcoin.  

Apparently they clear bitcoin trades?  Page 11 of the 2022 10K: "Customers can also use Cash App to invest their funds in U.S. listed stocks and exchange-traded funds ("ETFs") or buy and sell bitcoin"

Now that bitcoin has pulled back a bit (?) I was curious what SQ (Block) trades with now.

To that end I set up an objects file with a StockData class which does some data pulls using yfinance and munges stock returns into betas and R2 with a rolling window of 60 90 and 252 days.

The idea is to see how SQ correlates with different ETF / stocks over time to try to tease out what drives the stock today.

First I set up a list of peers:<br>
![image](https://user-images.githubusercontent.com/39496491/218769223-7cef902a-4779-4da5-8be1-d7e656fb50ab.png)

And then plot the relative return:<>br
As you can see, SQ had a nice bounce back in the heady bitcoin days but has underperformed its peers since.
<br>
![image](https://user-images.githubusercontent.com/39496491/218769536-f38db679-fc72-4b0c-ae8e-ead236fa9a1f.png)

I then used a list of ETF and single stocks as "factors" in a way to see if SQ still correlated with Bitcoin (using MSTR as a proxy) or if it traded with Oil (USO) or retail (AMZN) etc.<br>

![image](https://user-images.githubusercontent.com/39496491/218769865-f1fab971-5584-4c0f-b2b0-6a2b8be29a1c.png)


Or a better view is in a bar plot:<br>

![image](https://user-images.githubusercontent.com/39496491/218770338-d8a6d730-b037-4f6c-93e7-9b029bad19cc.png)

<br>
You can see how the R2 with GPN has been increasing over the past three windows 252, 90, and 60 day rolling.
<br>
To me, this suggests SQ is starting to trade more like a processor and less like a bitcoin stock, as you might imagine it would post the crypto selloff of 2022.
>>>>>>> cef3c2701d95b72a3bcca3df810b5baf05ff5e9e
# bank_assets_chart
# bank_assets_chart
