apt-get update
yes | apt-get install wget
wget --header="Host: dl.fbaipublicfiles.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3" --header="Accept-Language: en-US,en;q=0.9" --header="Referer: https://fasttext.cc/docs/en/english-vectors.html" --header="Cookie: __cfduid=d45739660988742965175c8b76e66c6431568654856" --header="Connection: keep-alive" "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip" -O "wiki-news-300d-1M.vec.zip" -c

yes | apt-get install zip
yes | apt-get install unzip

mkdir ../data/
unzip wiki-news-300d-1M.vec.zip -d ../data/
rm wiki-news-300d-1M.vec.zip