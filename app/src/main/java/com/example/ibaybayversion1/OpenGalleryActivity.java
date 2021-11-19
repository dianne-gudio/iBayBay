package com.example.ibaybayversion1;

import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

public class OpenGalleryActivity extends AppCompatActivity {
    private TextView yourImageText;
    private ImageView imagePreview;
    private Button selectBtn;

    BitmapDrawable drawable;
    Bitmap bitmap;
    String encodedImage = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_open_gallery);

        //Assign variables
        yourImageText = findViewById(R.id.yourImageText);
        imagePreview = findViewById(R.id.selectedImage);
        selectBtn = (Button) findViewById(R.id.selectBtn);

        //"Select Image" Button
        selectBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 100);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == 100){
            //Get the image that is selected
            Uri uri = data.getData();
            imagePreview.setImageURI(uri);

            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            } catch (IOException e) {
                e.printStackTrace();
            }

            //-----------------NOW ANALYZE BUTTON-----------------
            selectBtn.setText("Analyze");
            selectBtn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    drawable = (BitmapDrawable)imagePreview.getDrawable();
                    bitmap = drawable.getBitmap();
                    encodedImage = getStringImage(bitmap);

                    analyzeInput(encodedImage, uri);
                }
            });
        }
        yourImageText.setVisibility(View.INVISIBLE); //Hides "Your captured image will show here." message
    }

    private String getStringImage(Bitmap bitmap) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, baos);

        // Convert image to an encoded Base64 String
        byte[] imageBytes = baos.toByteArray();
        return Base64.encodeToString(imageBytes, Base64.DEFAULT);
    }

    public void analyzeInput(String imageInput, Uri imgUri){
        Python py = Python.getInstance();
        PyObject pyobj = py.getModule("baybay_caps_classify");
        PyObject pred1 = pyobj.callAttr("get_prediction", imageInput);

        GlobalVariables.classification = pred1.toString();
        GlobalVariables.imageInput = imgUri;

        showOutput();
    }

    public void showOutput(){
        Intent i = new Intent(this, OutputActivity.class); //Opens Output screen
        startActivity(i);
    }
}