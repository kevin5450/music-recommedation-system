package com.example.demo.model;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

@Document(collection = "user")
public class User {
    @Id
    private String id;

    private String userid;
    private String birthdate;
    private String email;
    private String password;
    private String nickname;
    private String fullname;
    private String phone;
    private boolean newsletter;

    // 기본 생성자
    public User() {}

    // 생성자
    public User(String userid, String fullname, String email, String password, String nickname, String birthdate, String phone, boolean newsletter) {
        this.userid = userid;
        this.fullname = fullname;
        this.email = email;
        this.password = password;
        this.nickname = nickname;
        this.birthdate = birthdate;
        this.phone = phone;
        this.newsletter = newsletter;
    }

    // Getter 및 Setter
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getUserid() { return userid; }
    public void setUserid(String userid) { this.userid = userid; }

    public String getNickname() { return nickname; }
    public void setNickname(String nickname) { this.nickname = nickname; }

    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }

    public String getPassword() { return password; }
    public void setPassword(String password) { this.password = password; }

    public boolean isNewsletter() { return newsletter; }
    public void setNewsletter(boolean newsletter) { this.newsletter = newsletter; }

    public String getFullname() { return fullname; }
    public void setFullname(String fullname) { this.fullname = fullname; }

    public String getBirthdate() { return birthdate; }
    public void setBirthdate(String birthdate) { this.birthdate = birthdate; }

    public String getPhone() { return phone; }
    public void setPhone(String phone) { this.phone = phone; }

}
