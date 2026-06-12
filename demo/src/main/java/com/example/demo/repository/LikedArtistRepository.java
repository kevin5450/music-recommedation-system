package com.example.demo.repository;

import com.example.demo.model.LikedArtist;
import org.springframework.data.mongodb.repository.MongoRepository;

import java.util.Optional;

public interface LikedArtistRepository extends MongoRepository<LikedArtist, String> {
    Optional<LikedArtist> findByArtist(String artist);
}
