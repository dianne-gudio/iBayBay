package com.example.ibaybayversion1;

import android.content.Intent;
import android.net.Uri;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

public class OpenGalleryActivity extends AppCompatActivity {
    private TextView yourImageText;
    private ImageView image;
    private Button selectBtn;

    int SELECT_IMAGE_CODE = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_open_gallery);

        //Assign variables
        yourImageText = findViewById(R.id.yourImageText);
        image = findViewById(R.id.selectedImage);
        selectBtn = (Button) findViewById(R.id.selectBtn);

        //"Select Image" Button
        selectBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent, "Title"), SELECT_IMAGE_CODE);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == 1){
            //Get the image that is selected
            Uri uri = data.getData();
            image.setImageURI(uri);

            //-----------------ANALYZE IMAGE-----------------TO BE EDITED---------
            selectBtn.setText("Analyze");
            selectBtn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    showOutput();
                }
            });
        }
        yourImageText.setVisibility(View.INVISIBLE); //Hides "Your captured image will show here." message

    }

    public void showOutput(){
        Intent i = new Intent(this, OutputActivity.class); //Opens Output screen
        startActivity(i);
    }
}