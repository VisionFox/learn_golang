package handler

type Trie struct {
	children [26]*Trie
	isEnd    bool
}

//func Constructor() Trie {
//	return Trie{}
//}

func (t *Trie) Insert(word string) {
	node := t
	for _, ch := range word {
		ch -= 'a'
		if node.children[ch] == nil {
			node.children[ch] = &Trie{}
		}
		node = node.children[ch]
	}
	node.isEnd = true
}

func (t *Trie) Search(word string) bool {
	node := t.searchPrefix(word)
	return node != nil && node.isEnd == true
}

func (t *Trie) StartsWith(prefix string) bool {
	return t.searchPrefix(prefix) != nil
}

func (t *Trie) searchPrefix(prefix string) *Trie {
	node := t
	for _, ch := range prefix {
		ch -= 'a'
		if node.children[ch] == nil {
			return nil
		}
		node = node.children[ch]
	}
	return node
}
