# gpt-text-split-and-converter
テキスト分割&分割したテキスト変換器  
コンテキスト長と戦うために分割方法をこねこねして遊んでみたリポジトリ  
今はテキスト校正ちゃんになっている  
API-Keyをセットしてスタート！

テキスト分割の優先度を
1. 改行(パラグラフ)
2. 句点（。|！|？|;）
3. token数

にしてみた。文脈をできるだけ保持したいなぁという気持ちだけど、ここら辺は自然言語のノウハウをしっかりと学んだ方が良さそう. 

あと、上記のつもりだけど実際にこう動いているかはちょっとわからない. 
動いてるっぽいからヨシ！  

やってみて思ったのがテストケースが作りにくい…  
曖昧な変換をお願いしているから、変換結果が妥当かどうかの設計が難しい…

きっと[LMQL](https://lmql.ai/)などと併せて柔軟なLLMを固くしていくのね

あと対話型UIは神  
あれはもしかしたら人力テストを都度挟んでるようなものなのかもしれない

今回は生PythonでLLMやってみたけど次からはLangChainとか使おうかな  
TextSplitterもあるし  
https://python.langchain.com/en/latest/reference/modules/text_splitter.html  
オーバーラップあるの良いな…  
https://note.com/npaka/n/nda9dc5eae1df
