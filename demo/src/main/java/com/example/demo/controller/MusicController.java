package com.example.demo.controller;

import com.example.demo.model.LikedArtist;
import com.example.demo.model.MyPlaylist;
import com.example.demo.model.Music;
import com.example.demo.repository.LikedArtistRepository;
import com.example.demo.repository.MyPlaylistRepository;
import com.example.demo.repository.MusicRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Optional;

@Controller
public class MusicController {

    @Autowired
    private MusicRepository songRepository;

    @Autowired
    private MyPlaylistRepository playlistRepository;

    @Autowired
    private LikedArtistRepository likedArtistRepository;

    @GetMapping("/recommendationpage")
    public String recommendationpage(Model model) {
        return "recommendationpage";
    }

    @GetMapping("/Settings")
    public String Settings(Model model) {
        return "Settings";
    }

    @PostMapping("/Mainpage")
    public String addSong(@RequestParam("title") String title,
                          @RequestParam("artist") String artist) {
        if (title == null || title.isEmpty() || artist == null || artist.isEmpty()) {
            return "redirect:/Mainpage";
        }

        Music song = new Music();
        song.setTitle(title);
        song.setArtist(artist);
        songRepository.save(song);

        return "redirect:/Mainpage";
    }

    @GetMapping("/search")
    public String searchSongs(@RequestParam(name = "query", required = false) String query, Model model) {
        List<Music> songs = query != null && !query.isEmpty()
                ? songRepository.findByTitleContainingOrArtistContainingIgnoreCase(query, query)
                : null;

        if (songs != null && !songs.isEmpty()) {
            model.addAttribute("songs", songs);
        } else {
            model.addAttribute("error", "No songs found");
        }
        return "Mainpage";
    }

    @PostMapping("/save")
    public String save(@RequestParam("title") String title, @RequestParam("artist") String artist) {
        Optional<MyPlaylist> existingSong = playlistRepository.findByTitleAndArtist(title, artist);

        if (existingSong.isEmpty()) {
            MyPlaylist newSong = new MyPlaylist(title, artist);
            playlistRepository.save(newSong);
        }

        return "redirect:/Mainpage";
    }

    @PostMapping("/save-artist")
    public String saveArtist(@RequestParam("artist") String artist) {
        Optional<LikedArtist> existingArtist = likedArtistRepository.findByArtist(artist);

        if (existingArtist.isEmpty()) {
            LikedArtist newArtist = new LikedArtist(artist);
            likedArtistRepository.save(newArtist);
        }

        return "redirect:/Mainpage";
    }

    @GetMapping("/Mainpage")
    public String MainpageLikedArtist(Model model) {
        List<LikedArtist> artists = likedArtistRepository.findAll();
        model.addAttribute("artists", artists);
        return "Mainpage";
    }
}
