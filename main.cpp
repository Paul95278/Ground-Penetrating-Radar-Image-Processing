//Ctrl+a,Ctrl+i 自动缩进
#include<iostream>
#include<fstream>
#include<iomanip>
#include<cstdlib>
#include<string>
#include<vector>
#include<cmath>
#include<opencv2/opencv.hpp>

struct DZTheader
{
    short tag;
    short header_size;
    short Sample_per_scan;
    short Bits_per_word;
    short binaryoffset;
    float Scan_per_sec;
    float Scan_per_meter;
    float Meters_per_mark;
    float Position;
    float Range;
    unsigned short Scans_per_pass;
    int createdate;
    int modifydate;
    unsigned short Offset_to_range_gain;
    unsigned short Size_of_range_gain;
    unsigned short Offset_to_text;
    unsigned short Size_of_text;
    unsigned short Offset_to_proc_hist;
    unsigned short Size_of_proc_hist;
    unsigned short Number_of_channels;
    float Dielectric_constant;
    float Top_pos_in_m;
    float Range_in_m;
    char reserved[31];
    char Data_type;
    char Antenna_name[14];
    unsigned short chan_mask;
    char This_file_name[12];
    short chksum;
    char variable[896];
};
struct DZT_inf
{
    int           ns;
    float         dx;
    float         sigpos;
    char          Antenna[14];
    short         TxRx;
    int           ntr;
    float         dt;
    std::vector<float>  tt2w;
    std::string         zlab;
    std::vector<float>  x;
    std::string         xlab;
};
struct time
{
    unsigned short sec;
    unsigned short mint;
    unsigned short hour;
    unsigned short day;
    unsigned short month;
    unsigned short year;
};

using namespace std;
using namespace cv;

void readheader(const char* file);
//定义结构体Result作为返回对象//注意： 结构体要定义在Result申明之前
struct ThreeValuedReturn
{
   Mat sanzhi;
   Mat value;
};
struct Result{
    Mat RectangleMark;
    Mat ThreeValuedProcessedSrc;
    Mat coord;
    Mat DiseaseSubMatrixData;
    Mat VIPDiseaseSubMatrixPic_ThreeValued[100];
    Mat DiseaseSubMatrixPic[100];
    float Threshold;
    int Sub_matrix_num;
    int disease_num;
    int VIP_disease_num;
    int NIP_disease_num;
    int Sub_matrix_rowlnum;
    int Sub_matrix_colnum;
    int sift_Threshold;
    int SrcRows;
    int SrcCols;
};
ThreeValuedReturn ThreeValuedProcess(Mat& src);
Mat draw_rectangle(Mat& src,Mat& coord,int BH_num);
void printResult(Result res);
Result split_row_variance(Mat& src,int pa);
void show_Mat_uchar(Mat& src);
void show_Mat_int(Mat& src);
void show_Mat_double(Mat& src);
Mat remove_average( Mat& src);
Mat trans32t8(Mat& src);
int same_value_unite(Mat& src);


Mat scale_depth_linear( Mat & src,int pa);
Mat scale_depth_exponent( Mat& src,int pa);
Mat scale_depth_compensation( Mat& src,int pa);

DZTheader hdr;
DZT_inf dzt_inf;
Mat data;
Mat cur_data;

Mat readdzt(const string& file)
{

    ifstream fin;
    fin.open(file.c_str(), ios_base::in | ios_base::binary);
    if (!fin.is_open())
    {
        cerr << "error:Fail to open the file";
        exit(1);
    }

    readheader(file.c_str());

    dzt_inf.ns = hdr.Sample_per_scan;
    if (hdr.Scan_per_meter)
        dzt_inf.dx = 1 / hdr.Scan_per_meter;
    dzt_inf.sigpos = hdr.Position;
    dzt_inf.TxRx = 0;
    fin.seekg(0, ios_base::end);
    long long sum_bytes = fin.tellg();
    long long data_offset;
    if(hdr.header_size<1024)
        data_offset=1024*hdr.header_size;
    else
        data_offset=1024;

    long long data_size = sum_bytes-data_offset ;
    fin.seekg(data_offset);


    if (hdr.Bits_per_word == 8)
    {
        unsigned char*data_inf1 = new unsigned char[data_size];
        dzt_inf.ntr = data_size / hdr.Sample_per_scan ;
        fin.read((char*)data_inf1, data_size);

        cout << "Now DZTdata have been imported!";
        Mat dztimg(dzt_inf.ntr, dzt_inf.ns, CV_8U, data_inf1);
        fin.close();
        dzt_inf.dt = hdr.Range / (hdr.Sample_per_scan - 1);
        for (float i = 0; i <= dzt_inf.dt*(hdr.Sample_per_scan - 1); i += dzt_inf.dt)
            dzt_inf.tt2w.push_back(i);
        dzt_inf.zlab = "Traveltime";
        if (hdr.Scan_per_meter == 0)
        {
            for (float i = 1; i <= dzt_inf.ntr; i++)
                dzt_inf.x.push_back(i);
            dzt_inf.xlab = "Scan Axis (Traces)";
        }
        else
        {
            for (float i = 0; i <= (dzt_inf.ntr - 1)*dzt_inf.dx; i += dzt_inf.dx)
                dzt_inf.x.push_back(i);
            dzt_inf.xlab = "Scan Axis(meters)";
        }

        return dztimg;

    }
    else if (hdr.Bits_per_word == 16)
    {
        unsigned short *data_inf2 = new unsigned short[data_size/2];
        int*data_int=new int[data_size/2];
        dzt_inf.ntr = (data_size / 2) / hdr.Sample_per_scan;
        fin.read((char*)data_inf2, data_size);

        cout << "Now DZTdata have been imported!" << endl;
        cout<<endl<<dzt_inf.ntr<<"*"<<dzt_inf.ns<<endl;

        for(int i=0;i<data_size/2;i++)
        {
            data_int[i]=data_inf2[i];
            data_int[i]-=32768;
        }
        delete[]data_inf2;


        Mat dzt_img(dzt_inf.ntr, dzt_inf.ns, CV_32SC1, data_int);
        for(int i=0;i<2;++i)
        {
            for(int j=0;j<dzt_img.rows;++j)
            {
                dzt_img.at<int>(j,i)=0;
            }
        }
        Mat dztimg=dzt_img.t();

        fin.close();
        return dztimg;
    }
    else if (hdr.Bits_per_word == 32)
    {
        int *data_inf3 = new  int[data_size/4];

        dzt_inf.ntr = (data_size / 4) / hdr.Sample_per_scan ;
        fin.read((char*)data_inf3, data_size );

        cout << "Now DZTdata have been imported!";
        cout<<endl<<"dzt_inf Size:"<<dzt_inf.ntr<<"*"<<dzt_inf.ns<<endl;


        Mat dzt_img(dzt_inf.ntr, dzt_inf.ns, CV_32SC1, data_inf3);
        for(int i=0;i<2;++i)
        {
            for(int j=0;j<dzt_img.rows;++j)
            {
                dzt_img.at<int>(j,i)=0;
            }
        }
        Mat dztimg=dzt_img.t();

        fin.close();
        return dztimg;
    }
}
void readheader(const char* file)
{
    extern DZTheader hdr;
    ifstream fin;
    fin.open(file, ios_base::in|ios_base::binary);
    short tb[5];
    float sr[5];
    unsigned short scanpp;
    int cm[2];
    unsigned short on[7];
    float dr[3];
    char resvdatp[32];
    char anname[14];
    unsigned short chan_mk;
    char This_f[12];
    short chksm;

    if (fin.is_open())
    {
        fin.seekg(0);
        fin.read((char*)(tb), 10);
        fin.read((char*)(sr), 20);
        fin.read((char*)(&scanpp), 2);
        fin.read((char*)(cm), 8);
        fin.read((char*)(on), 14);
        fin.read((char*)(dr),12);
        fin.read((char*)(resvdatp),32);
        fin.read((char*)(anname),14);
        fin.read((char*)(&chan_mk),2);

        fin.read((char*)(This_f),12);
        fin.read((char*)(&chksm),2);


        hdr.tag=tb[0];
        hdr.header_size=tb[1];
        hdr.Sample_per_scan=tb[2];
        hdr.Bits_per_word=tb[3];
        hdr.binaryoffset=tb[4];
        hdr.Scan_per_sec=sr[0];
        hdr.Scan_per_meter=sr[1];
        hdr.Meters_per_mark=sr[2];
        hdr.Position=sr[3];
        hdr.Range=sr[4];
        hdr.Scans_per_pass=scanpp;
        hdr.createdate=cm[0];
        hdr.modifydate=cm[1];
        hdr.Offset_to_range_gain=on[0];
        hdr.Size_of_range_gain=on[1];
        hdr.Offset_to_text=on[2];
        hdr.Size_of_text=on[3];
        hdr.Offset_to_proc_hist=on[4];
        hdr.Size_of_proc_hist=on[5];
        hdr.Number_of_channels=on[6];
        hdr.Dielectric_constant=dr[0];
        hdr.Top_pos_in_m=dr[1];
        hdr.Range_in_m=dr[2];
        for(int i=0;i<31;++i)
            hdr.reserved[i]=resvdatp[i];
        hdr.Data_type=resvdatp[31];
        for(int j=0;j<14;++j)
            hdr.Antenna_name[j]=anname[j];
        hdr.chan_mask=chan_mk;
        for(int i=0;i<12;++i)
            hdr.This_file_name[i]=This_f[i];
        hdr.chksum=chksm;
        //cout<<hdr.This_file_name<<endl;

    }
    else
    {
        cout << "error in reading header";
        exit(1);
    }
    fin.close();
}


int main(){
    string filename="E:\\GPR\\SJK_LYG\\First_plot_leftline\\FILE008-250-325.DZT";
    data=readdzt(filename);
    cur_data=data.clone();
    //data.rowRange(0,2)=32767;
    if (data.depth()==CV_32SC1){
        Result res=split_row_variance(data,374);
        printResult(res);
        show_Mat_int(res.coord);
        imwrite("C:\\Users\\pearson\\Desktop\\RectangleMark.bmp",res.RectangleMark);
        imwrite("C:\\Users\\pearson\\Desktop\\ThreeValuedProcessedSrc.bmp",res.ThreeValuedProcessedSrc);
        for(int i=0;i<res.VIP_disease_num;i++){
            string Img_Name0 = "C:\\Users\\pearson\\Desktop\\check\\" +to_string(i)+".bmp";
             imwrite(Img_Name0,res.VIPDiseaseSubMatrixPic_ThreeValued[i]);
        }


    }
         else{printf("ERROR FORMAT!");}

    return 0;
}

Result split_row_variance(Mat& src,int pa){
    struct Result disease;
    int rownum=src.rows,colnum=src.cols;
    int sub_matrix_num=colnum/pa;
    Mat sub_matrix(rownum,pa,CV_32SC1);
    Mat sub_norm(rownum,pa,CV_32SC1);
    Mat sub_matrix_norm_row(1,pa,CV_32SC1);
    Mat res(sub_matrix_num,rownum,CV_32SC1);
    int *ptmp=NULL;
    Mat src_norm;
    normalize(src,src_norm,255,0,NORM_MINMAX);
    for(int i =0;i<sub_matrix_num;i++)
    {
        ptmp = res.ptr<int>(i);
        if(i<sub_matrix_num)
        {
            src.colRange(i*pa,(i+1)*pa).copyTo(sub_matrix);}  //copyTo赋值数据 Mat A=B 不行
        else
        { src.colRange(colnum-pa,colnum).copyTo(sub_matrix);}

        //Mat xx;
        //sub_matrix.colRange(0,10).rowRange(0,10).copyTo(xx);
        //show_Mat_int(xx);
        normalize(sub_matrix,sub_norm,255,0,NORM_MINMAX);
        for (int j=0;j<rownum;j++)
        {
            Mat tep_m,tep_sd;
            sub_norm.row(j).copyTo(sub_matrix_norm_row);
            meanStdDev(sub_matrix_norm_row,tep_m,tep_sd);
            int variance=round(pow(tep_sd.at<double>(0,0),2));  //此处必须是double
            ptmp[j] = variance;
        }

    }
    transpose(res,res);

    //归一化0-255
    Mat res_normalize(rownum,sub_matrix_num,CV_32SC1);
    normalize(res,res_normalize,255,0,NORM_MINMAX);
    //show_Mat_int(res);

    Mat zhengtai(rownum*2,colnum,CV_32SC1);
    vconcat(res_normalize,-res_normalize,zhengtai);
    Mat tep_m0,tep_sd0;
    meanStdDev(zhengtai,tep_m0,tep_sd0);
    float thresh=tep_sd0.at<double>(0,0)*sqrt(2);
   // printf("\nTHRESHOLD: %f\n" ,thresh);

    double MaxVal;
    //找出存在＞阈值的行方差的列，即子图，BH：病害
    int BH_num=0,shangx,xiax=0;
    printf("*Sub_plots Exists BH:\n");
    for (int i=0;i<sub_matrix_num;i++){
        minMaxLoc(res_normalize.col(i),0,&MaxVal,0,0);
        //minMaxLoc(temp, &minVal, &maxVal, &minLoc, &maxLoc);
        if (MaxVal>thresh){
            //printf("%4d ",i+1);
            BH_num++;
        }
    }
    //if (BH_num==0){printf("Without BH");return res_normalize;}
    printf("\n");
    //保存病害子图的序号
    //注意：BH_list里面的基数从1开始
    int BH_list[BH_num];
    //int BH_data[BH_num][5];
    //**************************
    //**************************
    Mat BH_data(BH_num,8,CV_32SC1,Scalar(0));
    for (int i=1,j=0;i<sub_matrix_num+1;i++){
        minMaxLoc(res_normalize.col(i-1),0,&MaxVal,0,0);
        if (MaxVal>thresh){
            BH_list[j++]=i;
        }
    }

    for(int i=0;i<BH_num;i++){
        float percen=0.0;
        int BH_rows=0;
        //BH子图BH行数
        for(int j=0;j<rownum;j++){
            if(res_normalize.at<int>(j,BH_list[i]-1)>thresh){
                BH_rows++;
            }
        }
        //从上往下找出第一个大于阈值的行方差的位置shangx
        for(int j=0;j<rownum;j++){
            if(res_normalize.at<int>(j,BH_list[i]-1)>thresh){
                shangx=j;
                break;
            }
        }

        //从下往上找出最后一个大于阈值的行方差的位置xiax
        for(int j=rownum-1;j>-1;j--){
            if(res_normalize.at<int>(j,BH_list[i]-1)>thresh){
                xiax=j;
                break;
            }

        }
        percen=BH_rows*1.0/rownum;
        //printf("%4d%4d%5d  %f\n",shangx,xiax,BH_rows,percen);
        //BH_data 使用说明：
        //BH_data[i][0]=BH_list[i];  //BH子图序号+1
        BH_data.at<int>(i,0)=BH_list[i];
        BH_data.at<int>(i,1)=shangx;
        BH_data.at<int>(i,2)=xiax;
        BH_data.at<int>(i,3)=BH_rows;
        BH_data.at<int>(i,4)=round(percen*100);

    }

    //保存有病害子图
    for(int i=0;i<BH_num;i++){
        string Img_Name0 = "C:\\Users\\pearson\\Desktop\\Result\\" +to_string(BH_list[i])+".bmp";
        imwrite(Img_Name0,src_norm.colRange((BH_list[i]-1)*pa,BH_list[i]*pa).\
                rowRange(BH_data.at<int>(i,1),BH_data.at<int>(i,2)));}
    //有病害子图传入Result//***********************************
    for(int i=0;i<BH_num;i++){
        src_norm.colRange((BH_list[i]-1)*pa,BH_list[i]*pa).\
                rowRange(BH_data.at<int>(i,1),BH_data.at<int>(i,2)).copyTo(disease.DiseaseSubMatrixPic[i]);
    }
    //原图预处理，并扩展
Mat BH_submatrix;
//筛选出VIP病害子图
int sift_thre=10;
int VIP_num=0;
Mat coord(BH_num,5,CV_32SC1,Scalar(0));
for(int i=0;i<BH_num;i++){
    if(BH_data.at<int>(i,4)>=sift_thre){
    coord.at<int>(i,4)=1;
    src.colRange((BH_list[i]-1)*pa,(BH_list[i])*pa+1).rowRange(BH_data.at<int>(i,1)-20,BH_data.at<int>(i,2)+1).copyTo(BH_submatrix);
    struct ThreeValuedReturn data=ThreeValuedProcess(BH_submatrix);
    Mat sanzhi=data.sanzhi;
    BH_data.at<int>(i,6)=data.value.at<int>(0,0);
    BH_data.at<int>(i,7)=data.value.at<int>(0,1);
    //*************************************
    /*
    Mat srcReAv(BH_submatrix.size(),CV_32SC1);
    remove_average(BH_submatrix).copyTo(srcReAv);
    normalize(srcReAv,srcReAv,255,0,NORM_MINMAX);
    //32bit图转换为8bit
    Mat srcReAv8=trans32t8(srcReAv);
    Mat srcReAv8eq;
    equalizeHist(srcReAv8,srcReAv8eq);
    meanStdDev(srcReAv8eq,sub_mean,sub_std);
    //printf("%8.1f%8.1f\n",sub_mean.at<double>(0,0),sub_std.at<double>(0,0));

    //三值化上下阈值
    shangY=round(sub_mean.at<double>(0,0)+sub_std.at<double>(0,0));
    BH_data.at<int>(i,6)=shangY;
    xiaY=round(sub_mean.at<double>(0,0)-sub_std.at<double>(0,0));
    BH_data.at<int>(i,7)=xiaY;
    **************************************************************
    string Img_Name1 = "C:\\Users\\pearson\\Desktop\\Result\\srcReAv8eq_" +to_string(BH_list[i])+".bmp";
    imwrite(Img_Name1,srcReAv8eq);
    //寻找概率最大灰度
    Mat gailv(1,256,CV_32SC1,Scalar(0));
    for(int i=0;i<srcReAv8eq.rows;i++){
        int* ptp=gailv.ptr<int>(0);
        uchar* ptr=srcReAv8eq.ptr<uchar>(i);
        for (int j=0;j<srcReAv8eq.cols;j++){
            ptp[ptr[j]]++;
        }
    }
    Point maxgrey;
    uchar M_gailv_grey;
    minMaxLoc(gailv,0,0,0,&maxgrey);
    M_gailv_grey=uchar(maxgrey.x);
    //printf("%d",M_gailv_grey);

    //三值化
    Mat sanzhi=srcReAv8eq.clone();
    for(int i=0;i<srcReAv8eq.rows;i++){
        uchar* ptr=sanzhi.ptr<uchar>(i);
        for (int j=0;j<sanzhi.cols;j++){
            if(ptr[j]<=xiaY){
                ptr[j]=0;
            }
            else if(ptr[j]>=shangY){
                ptr[j]=255;
            }
            else{
                ptr[j]=M_gailv_grey;
            }
        }
    }*/
    //保存三值化有病害子图
    string Img_Name2 = "C:\\Users\\pearson\\Desktop\\Result\\sanzhi_" +to_string(BH_list[i])+".bmp";
    imwrite(Img_Name2,sanzhi);
    //VIP病害三值化子图传入Result//***********************************
    sanzhi.copyTo(disease.VIPDiseaseSubMatrixPic_ThreeValued[VIP_num]);
    //向下求导
    //求导结果三值化（元素值小于0的记为-1，元素值大于0的记为1）
    Mat sanzhi_qiudao(BH_submatrix.rows-1,BH_submatrix.cols,CV_32SC1);

    for(int i=1;i<BH_submatrix.rows;i++){
        uchar* ptr_qiudao_0=sanzhi.ptr<uchar>(i-1);
        uchar* ptr_qiudao_1=sanzhi.ptr<uchar>(i);
        int* ptr=sanzhi_qiudao.ptr<int>(i-1);
        for (int j=0;j<BH_submatrix.cols;j++){
            int x=ptr_qiudao_1[j]-ptr_qiudao_0[j];
            if(x>0) ptr[j]=1;
            else if(x<0) ptr[j]=-1;
            else ptr[j]=0;
            }
        }
     int phase=same_value_unite(sanzhi_qiudao);

     BH_data.at<int>(i,5)=phase;  
     VIP_num++;
    }
}
    //计算出病害区域左上角、右下角坐标
    for(int i=0;i<BH_num;i++){
        int* ptr=coord.ptr<int>(i);
        ptr[0]=(BH_data.at<int>(i,0)-1)*pa;
        ptr[1]=BH_data.at<int>(i,1);
        ptr[2]=BH_data.at<int>(i,0)*(pa-1);
        ptr[3]=BH_data.at<int>(i,2);
    }

Mat ThreeValuedProcessedSrc=ThreeValuedProcess(src).sanzhi;
Mat RectangleMark=draw_rectangle(ThreeValuedProcessedSrc,coord,BH_num);

    //待返回结构体赋值
    RectangleMark.copyTo(disease.RectangleMark);
    ThreeValuedProcessedSrc.copyTo(disease.ThreeValuedProcessedSrc);
    BH_data.copyTo(disease.DiseaseSubMatrixData);
    coord.copyTo(disease.coord);
    disease.Threshold=thresh;
    disease.disease_num=BH_num;
    disease.VIP_disease_num=VIP_num;
    disease.NIP_disease_num=BH_num-VIP_num;
    disease.Sub_matrix_num=sub_matrix_num;
    disease.Sub_matrix_colnum=pa;
    disease.Sub_matrix_rowlnum=rownum;
    disease.sift_Threshold=sift_thre;
    disease.SrcRows=src.rows;
    disease.SrcCols=src.cols;

    return disease;

}


int same_value_unite(Mat& src){
    int row=src.rows;
    int col=src.cols;
    Mat srct(col,row,CV_32SC1);
    srct=src.t();
    //int rowt=srct.rows;
    //int colt=srct.cols;
    Mat res(srct.rows,srct.cols,CV_32SC1);
    for(int i=0;i<srct.rows;i++){
        int notzero=0;
        int fill=1;
        int* ptr=srct.ptr<int>(i);
        for(int j=0;j<srct.cols;j++){
            if(ptr[j]!=0) notzero++;
            if(j==0) res.at<int>(i,fill++)=ptr[j];
            else{
                if(ptr[j]==ptr[j-1]) continue;
                else res.at<int>(i,fill++)=ptr[j];
            }
        }if(notzero>=6) res.at<int>(i,0)=1;
    }
    int pos=0;
    int neg=0;
    for(int i=0;i<res.rows;i++){
        if(res.at<int>(i,0)>0){
            for(int j=1;j<res.cols;j++){
                if(res.at<int>(i,j)==1)
                    {pos++;break;}
                else if(res.at<int>(i,j)==-1)
                {neg++;break;}
                else{;}

            }
        }
    }
    //show_Mat_int(res);
    //printf("%d\n%d",pos,neg);
    int phase=0;
    if(pos>neg) phase=1;
    else if(pos<neg) phase=-1;
    return phase;

}
void show_Mat_uchar(Mat& src){
    int rownum=src.rows,colnum=src.cols;
    for(int i=0;i<rownum;i++){
        uchar* data=src.ptr<uchar>(i);
        for (int j=0;j<colnum;j++){
            printf("%8d ",data[j]);
        }
        printf("\n");
    }
}
void show_Mat_int(Mat& src){
    int rownum=src.rows,colnum=src.cols;
    for(int i=0;i<rownum;i++){
        int* data=src.ptr<int>(i);
        for (int j=0;j<colnum;j++){
            printf("%8d ",data[j]);
        }
        printf("\n");
    }
}
void show_Mat_double(Mat& src){
    int rownum=src.rows,colnum=src.cols;
    for(int i=0;i<rownum;i++){
        double* data=src.ptr<double>(i);
        for (int j=0;j<colnum;j++){
            printf("%6.1lf ",data[j]);
        }
        printf("\n");
    }
}
Mat trans32t8(Mat& src){
    int rownum=src.rows,colnum=src.cols;
    Mat src8(src.size(),CV_8UC1);
    for (int i=0;i<rownum;i++){
        int* ptr32=src.ptr<int>(i);
        uchar* ptr8=src8.ptr<uchar>(i);
        for (int j=0;j<colnum;j++){
            ptr8[j]=static_cast<uchar>(ptr32[j]);
        }
    }
    return src8;
}
Mat remove_average(Mat &src)
{
        Mat ave_remove(src.size(),CV_32SC1);
        double average[src.rows];
        double sum=0;
        for(int x=0;x<src.rows;x++)
        {
            int *pr=src.ptr<int>(x);

            for(int y=0;y<src.cols;y++)
            {

                sum+=(double)pr[y];


            }
            average[x]=sum*1.0/src.cols;
            sum=0;
        }


        for(int i=0;i<src.rows;++i)
        {
            int *pt=src.ptr<int>(i);
            int *pd=ave_remove.ptr<int>(i);
            for(int j=0;j<src.cols;++j)
            {
                pd[j]=round(pt[j]-average[i]);
            }
        }
        return ave_remove;
    }
void printResult(Result res){
    show_Mat_int(res.DiseaseSubMatrixData);
    printf("\n");
    printf("SrcRows             :  %-8d\n",res.SrcRows);
    printf("SrcCols             :  %-8d\n",res.SrcCols);
    printf("Threshold           :  %-8.2f\n",res.Threshold);
    printf("Sub_matrix_num      :  %-8d\n",res.Sub_matrix_num);
    printf("disease_num         :  %-8d\n",res.disease_num);
    printf("VIP_disease_num     :  %-8d\n",res.VIP_disease_num);
    printf("NIP_disease_num     :  %-8d\n",res.NIP_disease_num); 
    printf("Sub_matrix_rowlnum  :  %-8d\n",res.Sub_matrix_rowlnum);
    printf("Sub_matrix_colnum   :  %-8d\n",res.Sub_matrix_colnum);
    printf("sift_Threshold      :  %-8d\n",res.sift_Threshold);
}

Mat draw_rectangle(Mat& src,Mat& coord,int BH_num){
    Mat dst;
    cvtColor(src, dst, COLOR_GRAY2RGB );
    //CvFont font;
    //cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.5f, 1.5f, 0, 2, CV_AA);//设置显示的字体
    //char *strID;
    //strID="MyObject";
    Point P0,P1;
    for(int i=0;i<BH_num;i++){
         int* ptr=coord.ptr<int>(i);
         P0.x=ptr[0];
         P0.y=ptr[1]-10;
         P1.x=ptr[2];
         P1.y=ptr[3];
         if(ptr[4]==1)  rectangle(dst,P0 ,P1, CV_RGB(255, 0, 0), 2);
         else           rectangle(dst,P0 ,P1, CV_RGB(0, 255, 0), 2);	//绿色画框
         //putText(dst,strID, Point(P0.x, P0.y-10), &font, CV_RGB(255, 0, 0));
    }
    return dst;
}
ThreeValuedReturn ThreeValuedProcess(Mat& src){
    //src CV_32SC1
    struct ThreeValuedReturn data;
    Mat sub_mean,sub_std;
    Mat srcReAv(src.rows,src.cols,CV_32SC1);
    remove_average(src).copyTo(srcReAv);
    normalize(srcReAv,srcReAv,255,0,NORM_MINMAX);
    //32bit图转换为8bit
    Mat srcReAv8=trans32t8(srcReAv);
    Mat srcReAv8eq;
    equalizeHist(srcReAv8,srcReAv8eq);
    meanStdDev(srcReAv8eq,sub_mean,sub_std);
    //三值化上下阈值
    int shangY=round(sub_mean.at<double>(0,0)+sub_std.at<double>(0,0));
    int xiaY=round(sub_mean.at<double>(0,0)-sub_std.at<double>(0,0));
    Mat value(1,2,CV_32SC1);
    Mat sanzhi=srcReAv8eq.clone();
    for(int i=0;i<srcReAv8eq.rows;i++){
        uchar* ptr=sanzhi.ptr<uchar>(i);
        for (int j=0;j<sanzhi.cols;j++){
            if(ptr[j]<=xiaY){
                ptr[j]=0;
            }
            else if(ptr[j]>=shangY){
                ptr[j]=255;
            }
            else{
                ptr[j]=127;
            }
        }
    }
    value.at<int>(0,0)=xiaY;
    value.at<int>(0,1)=shangY;
    data.sanzhi=sanzhi;
    value.copyTo(data.value);
    return data;
}
