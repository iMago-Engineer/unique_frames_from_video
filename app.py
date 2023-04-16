import hashlib
import streamlit as st
import shutil
from main import *
from io import BytesIO
from zipfile import ZipFile

TEMP_DIR = 'temp'
UPLOAD_DIR = f'{TEMP_DIR}/uploads'
DOWNLOAD_DIR = f'{TEMP_DIR}/downloads'

NUM_COLS = 5

def save_uploaded_file(uploaded_file):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    destination_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(destination_path, "wb") as f:
        data = uploaded_file.getbuffer()
        f.write(data)

        # md5 = hashlib.md5()
        # md5.update(data)
        # print(md5.hexdigest())

    return destination_path

def main():
    st.set_page_config(page_title='Video Processing App')
    st.title('Video Processing App')

    # 動画フィアルのアップロード
    uploaded_file  = st.file_uploader("Choose a video file (.mp4)", type="mp4")

    if uploaded_file is not None:
        # 動画ファイルを OpenCV で読み込むために、一回保存しておく必要がある
        uploaded_file_path = save_uploaded_file(uploaded_file)

        frames = read_frames_from_video_file(uploaded_file_path)

        frames_with_diff_edges = extract_frames_with_diff_edges(frames)
        distinct_frames = remove_similar_frames(frames_with_diff_edges)

        # For DEBUG
        # output_dir = 'output'
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # for i, frame in enumerate(distinct_frames):
        #     save_frame_as_image(output_dir, frame, i)

        frames_selected = False
        selected_frames = []

        st.header("抽出画面")
        with st.form("画面"):
            cols = st.columns(NUM_COLS)

            for i, frame in enumerate(distinct_frames):
                with cols[i % NUM_COLS]:
                    with st.container():
                        st.image(frame, use_column_width=True, channels='BGR')
                        if st.checkbox("👆", key=f'frame_{i}'):
                            selected_frames.append(frame)

            frames_selected = st.form_submit_button("画像ダウンロードの準備")

        # 画面の選択が終わったら、ダウンロードボタンを表示する
        if frames_selected:
            if len(selected_frames) > 0:
                zip_buffer = zip_images(selected_frames)
                st.download_button(label='⬇️ 選択した画面をまとめてダウンロード', data=zip_buffer, file_name='frames.zip', mime='application/zip')

        cv2.destroyAllWindows()

@st.cache_data
def zip_images(frames):
    # Create a temporary directory to store the frames
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    # Save the frames as JPEG files in the temporary directory
    for i, frame in enumerate(frames):
        file_path = os.path.join(DOWNLOAD_DIR, f'frame_{i}.jpg')
        cv2.imwrite(file_path, frame)

    # Compress the files in the temporary directory as a ZIP archive
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, 'w') as zip_file:
        for file_name in os.listdir(DOWNLOAD_DIR):
            file_path = os.path.join(DOWNLOAD_DIR, file_name)
            zip_file.write(file_path, file_name)

    shutil.rmtree(DOWNLOAD_DIR)

    return zip_buffer.getvalue()

if __name__ == '__main__':
    main()
