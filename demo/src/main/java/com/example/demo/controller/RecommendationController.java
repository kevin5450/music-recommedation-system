package com.example.demo.controller;

import com.example.demo.model.RecommendationPersonalPage;
import com.example.demo.model.RecommendationPlaylist;
import com.example.demo.model.RecommendationUnexpected;
import com.example.demo.repository.RecommendationPersonalPageRepository;
import com.example.demo.repository.RecommendationPlaylistRepository;
import com.example.demo.repository.RecommendationUnexpectedRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.List;


@Controller
public class RecommendationController {

    @Autowired
    private RecommendationPlaylistRepository recommendationPlaylistRepository;

    @Autowired
    private RecommendationPersonalPageRepository recommendationPersonalPageRepository;

    @Autowired
    private RecommendationUnexpectedRepository recommendationUnexpectedRepository;

    @GetMapping("/recommendation-personalpage")
    public String recommendationPersonalPage(Model model) {
        List<RecommendationPersonalPage> personalPageList = recommendationPersonalPageRepository.findAll();
        model.addAttribute("personalPageList", personalPageList);
        return "recommendation-personalpage"; // templates/recommendation-personalpage.html
    }

    @GetMapping("/recommendation-playlist")
    public String recommendationPlaylist(Model model) {
        List<RecommendationPlaylist> playlistList = recommendationPlaylistRepository.findAll();
        model.addAttribute("playlistList", playlistList);
        return "recommendation-playlist"; // templates/recommendation-playlist.html
    }

    @GetMapping("/recommendation-unexpected")
    public String recommendationUnexpected(Model model) {
        List<RecommendationUnexpected> unexpectedList = recommendationUnexpectedRepository.findAll();
        model.addAttribute("unexpectedList", unexpectedList);
        return "recommendation-unexpected"; // templates/recommendation-unexpected.html
    }
}

