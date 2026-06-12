package com.example.demo.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

@RestController
public class RecommendProxyController {

    @GetMapping("/recommend")
    public ResponseEntity<String> proxyToFlask(@RequestParam String query) {
        String flaskUrl = "http://localhost:5000/recommend?query=" + URLEncoder.encode(query, StandardCharsets.UTF_8);
        RestTemplate restTemplate = new RestTemplate();
        return restTemplate.getForEntity(flaskUrl, String.class);
    }
}
