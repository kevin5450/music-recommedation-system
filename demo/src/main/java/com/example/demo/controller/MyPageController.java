package com.example.demo.controller;

import com.example.demo.model.LikedArtist;
import com.example.demo.repository.LikedArtistRepository;
import com.example.demo.repository.MyPlaylistRepository;
import com.example.demo.model.MyPlaylist;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

import java.util.List;

@Controller
public class MyPageController {

    @Autowired
    private MyPlaylistRepository playlistRepository;

    @Autowired
    private LikedArtistRepository likedArtistRepository;

    @GetMapping("/")
    public String showloginPage() {
        return "login";
    }

    @GetMapping("/Mypage")
    public String MyPage(Model model) {
        return "Mypage";
    }

    @GetMapping("/mypage-myplaylist")
    public String showPlaylist(Model model) {
        List<MyPlaylist> playlist = playlistRepository.findAll();
        model.addAttribute("playlist", playlist);
        return "mypage-myplaylist";
    }

    @GetMapping("/mypage-likedartist")
    public String MyPageLikedArtist(Model model) {
        List<LikedArtist> artists = likedArtistRepository.findAll();
        model.addAttribute("artists", artists);
        return "mypage-likedartist";
    }
}