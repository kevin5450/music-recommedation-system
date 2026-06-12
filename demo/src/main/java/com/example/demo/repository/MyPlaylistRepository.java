package com.example.demo.repository;

import com.example.demo.model.MyPlaylist;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface MyPlaylistRepository extends MongoRepository<MyPlaylist, String> {
    Optional<MyPlaylist> findByTitleAndArtist(String title, String artist);
}
