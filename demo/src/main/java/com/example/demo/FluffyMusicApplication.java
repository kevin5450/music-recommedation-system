package com.example.demo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication(scanBasePackages = "com.example.demo")
public class FluffyMusicApplication {

    public static void main(String[] args) {
        System.out.println("Start======================================");
        SpringApplication.run(FluffyMusicApplication.class, args);

    }
}

