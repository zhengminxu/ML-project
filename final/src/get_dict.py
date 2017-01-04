from util import get_token, clean_html
data = pd.read_csv("../data/test.csv")
data = data.content.values.tolist()
data = clean_html(data)
data = [re.sub(r'\n', ' ', x) for x in data]
data = [get_token(x) for x in data]
