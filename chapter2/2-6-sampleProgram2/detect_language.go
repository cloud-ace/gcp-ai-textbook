package main

import (
	"context"
	"fmt"
	"io/ioutil"
	"log"

	"cloud.google.com/go/translate"
)

const (
	filePathJa = "../constitution_of_japan_ja.txt"
	filePathEn = "../constitution_of_japan_en.txt"
)

func main() {
	ctx := context.Background()

	client, err := translate.NewClient(ctx)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	txtJa, err := ioutil.ReadFile(filePathJa)
	if err != nil {
		log.Fatalf("Failed to read file of ja: %v", err)
	}

	txtEn, err := ioutil.ReadFile(filePathEn)
	if err != nil {
		log.Fatalf("Failed to read file of en: %v", err)
	}

	// 1
	inputs := []string{
		string(txtJa),
		string(txtEn),
	}

	// 2
	detections, err := client.DetectLanguage(ctx, inputs)
	if err != nil {
		log.Fatalf("Failed to detect languages: %v", err)
	}

	// 3
	for _, detection := range detections {
		// 4
		for _, result := range detection {
			// 5
			fmt.Printf("検出言語: %v, 確度: %v\n", result.Language, result.Confidence)
		}
	}
}
