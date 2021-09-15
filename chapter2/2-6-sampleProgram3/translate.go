package main

import (
	"context"
	"fmt"
	"io/ioutil"
	"log"
	"strings"

	"cloud.google.com/go/translate"
	"golang.org/x/text/language"
)

const (
	filePathEn = "../constitution_of_japan_en.txt"
)

func main() {
	ctx := context.Background()

	client, err := translate.NewClient(ctx)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	txtEn, err := ioutil.ReadFile(filePathEn)
	if err != nil {
		log.Fatalf("Failed to read file of en: %v", err)
	}

	// 1
	var inputs []string
	for _, line := range strings.Split(string(txtEn), "\n") {
		sentences := strings.SplitAfter(line, ".")
		for _, sentence := range sentences {
			if sentence != "" {
				inputs = append(inputs, strings.TrimSpace(sentence))
			}
		}
	}

	// 2
	ts, err := client.Translate(ctx, inputs, language.Japanese, nil)
	if err != nil {
		log.Fatalf("Failed to detect languages: %v", err)
	}

	// 3
	for _, t := range ts {
		fmt.Printf("%s\n", t.Text)
	}
}
