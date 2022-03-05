package solution

import "sort"

func Solution() {
	println("hello")
}

func FindContentChildren(g []int, s []int) int {
	sort.Ints(g)
	sort.Ints(s)
	indexG, indexS := 0, 0

	for indexG < len(g) && indexS < len(s) {
		if g[indexG] <= s[indexS] {
			indexG++
		}
		indexS++
	}
	return indexG
}
