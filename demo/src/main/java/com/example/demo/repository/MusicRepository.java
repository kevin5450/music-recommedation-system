package com.example.demo.repository;

import com.example.demo.model.Music;
import org.springframework.data.mongodb.repository.MongoRepository;

import java.util.List;

public interface MusicRepository extends MongoRepository<Music, String> {

    List<Music> findByTitleContainingOrArtistContainingIgnoreCase(String query, String query1);

    List<Music> findByPopularityGreaterThanEqual(int i);
}
// SongRepository는 데이터베이스와 상호작용을 하는 레파지토리 인터페이스입니다
// repository는 데이터 저장 및 검색을 처리하는데 사용
// SongRepository는 노래 데이터를 몽고디비에 저장하고 불러오는 기능을 제공