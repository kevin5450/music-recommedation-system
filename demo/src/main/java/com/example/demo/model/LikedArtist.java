package com.example.demo.model;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

@Document(collection = "liked_artist")  // MongoDB 컬렉션 이름
public class LikedArtist {
    @Id
    private String id;  // MongoDB의 기본 키 (_id)
    private String artist;  // 가수 이름 저장

    // 기본 생성자 (MongoDB용)
    public LikedArtist() {}

    // 생성자
    public LikedArtist(String artist) {
        this.artist = artist;
    }

    // Getter & Setter
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getArtist() { return artist; }
    public void setArtist(String artist) { this.artist = artist; }
}
