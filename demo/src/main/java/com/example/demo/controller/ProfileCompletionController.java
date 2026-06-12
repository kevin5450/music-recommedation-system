package com.example.demo.controller;

import com.example.demo.model.Music;
import com.example.demo.model.FirstFiveSongs;
import com.example.demo.repository.MusicRepository;
import com.example.demo.repository.FirstFiveSongsRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

@Controller
public class ProfileCompletionController {

    @Autowired
    private MusicRepository musicRepository;

    @Autowired
    private FirstFiveSongsRepository firstFiveSongsRepository;

    /**
     * 프로필 완료 페이지 - MongoDB에서 인기 있는 노래를 가져와서 랜덤으로 30곡 표시
     */
    @GetMapping("/profile-completion")
    public String showProfileCompletionPage(Model model) {
        // MongoDB에서 popularity가 60 이상인 곡 조회
        List<Music> allPopularSongs = musicRepository.findByPopularityGreaterThanEqual(60);

        // 리스트를 섞고 상위 30곡 가져오기
        Collections.shuffle(allPopularSongs);
        List<Music> randomSongs = allPopularSongs.subList(0, Math.min(30, allPopularSongs.size()));

        model.addAttribute("songs", randomSongs);
        return "profile-completion";
    }

    /**
     * 유저가 선택한 5곡 저장 또는 자동 선택 후 저장
     */
    @PostMapping("/save-first-five")
    public String saveFirstFiveSongs(
            @RequestParam(value = "selectedSongs", required = false) List<String> selectedSongIds,
            Model model) {

        // 유저가 5곡을 선택하지 않은 경우, MongoDB에서 랜덤으로 5곡 가져오기
        if (selectedSongIds == null || selectedSongIds.size() < 5) {
            List<Music> allPopularSongs = musicRepository.findByPopularityGreaterThanEqual(60);
            Collections.shuffle(allPopularSongs);
            List<Music> autoSelectedSongs = allPopularSongs.subList(0, Math.min(5, allPopularSongs.size()));

            // 자동으로 선택한 곡 ID 리스트 생성
            selectedSongIds = autoSelectedSongs.stream()
                    .map(Music::getId)
                    .collect(Collectors.toList());
        }

        // 선택된 5곡을 FirstFiveSongs 테이블에 저장
        for (String songId : selectedSongIds) {
            Music selectedSong = musicRepository.findById(songId).orElse(null);
            if (selectedSong != null) {
                FirstFiveSongs firstSong = new FirstFiveSongs(selectedSong.getTitle(), selectedSong.getArtist());
                firstFiveSongsRepository.save(firstSong);
            }
        }

        return "redirect:/Mainpage";
    }
}
