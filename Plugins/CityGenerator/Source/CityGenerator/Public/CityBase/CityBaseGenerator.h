// Copyright ChengWei, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "CityBaseGenerator.generated.h"

class UMaterialInterface;

USTRUCT(BlueprintType, DisplayName = "双线合并参数")
struct CITYGENERATOR_API FMergeTwoLaneParams
{
	GENERATED_BODY()
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen")
	float MinJunctionAngle = 30.0;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen")
	float SearchRadius = 30.0;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen")
	float MaxMergeAngle = 25.0;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen")
	float MinDistToEnd = 5.0;
};

USTRUCT(BlueprintType)
struct CITYGENERATOR_API FRoadParams
{
	GENERATED_BODY()

	/// <summary>
	/// 车道宽度
	/// </summary>
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen", DisplayName = "车道宽度")
	float LandWidth = 3.0;

	/// <summary>
	/// 道路厚度
	/// </summary>
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen", DisplayName = "道路厚度")
	float Thickness = 0.2;

	/// <summary>
	/// 路口细分点数
	/// </summary>
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen", DisplayName = "路口细分点数")
	int32 SegmentNum = 10;

	/// <summary>
	/// 是否生成标线
	/// </summary>
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen", DisplayName = "是否生成标线")
	bool bNeedTrafficLine = false;

	/// <summary>
	/// 样条采样时的每段长度
	/// </summary>
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen", DisplayName = "样条采样时的每段长度")
	float SegmentLength = 1;

	/// <summary>
	/// 是否需要双线合并
	/// </summary>
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen", DisplayName = "是否需要双线合并")
	bool bMergeTwoLine = true;

	/// <summary>
	/// 只有开启双线合并后，该参数才起作用
	/// </summary>
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen", DisplayName = "双线合并参数")
	FMergeTwoLaneParams MergeTwoLaneParams;

	/// <summary>
	/// 是否需要高程匹配
	/// </summary>
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen", DisplayName = "是否需要高程匹配")
	bool bHeightMatch = true;

	FString TerrainTifPath;
};

USTRUCT(BlueprintType)
struct CITYGENERATOR_API FRoadOutContext
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Road")
	FString LightPosContent;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Road")
	FString FacilityContent;
};

USTRUCT(BlueprintType)
struct CITYGENERATOR_API FElevationMatchParams
{
	GENERATED_BODY()
	
	FString BoundsFilePath;
	
	FString BuildingFilePath;
	
	FString RoadFilePath;

	FString GreenFilePath;
	
	FString WaterFilePath;
	
	FString GroundFilePath;
	
	FString BlockFilePath;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen", DisplayName = "水平面高度")
	float WaterOffsetAlt=-1;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen", DisplayName = "地块高度")
	float BlockHeight = 0.2;
};

USTRUCT(BlueprintType)
struct CITYGENERATOR_API FTerrainParams
{
	GENERATED_BODY()

	/// <summary>
	/// 是否需要高程匹配
	/// </summary>
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen", DisplayName = "是否需要高程匹配")
	bool bElevationMatch = true;
	/// <summary>
	/// 只有开启高程匹配后，该参数才起作用
	/// </summary>
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen", DisplayName = "高程匹配参数")
	FElevationMatchParams ElevationMatchParams;
};

UCLASS()
class CITYGENERATOR_API ACityBaseGenerator : public AActor
{
	GENERATED_BODY()

public:
	// Sets default values for this actor's properties
	ACityBaseGenerator();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	/// <summary>
	/// 道路生成
	/// </summary>
	/// <param name="RoadPath">道路文件绝对路径</param>
	/// <param name="RoadParams"></param>
	UFUNCTION(BlueprintCallable, Category = "CityGen")
	FRoadOutContext GenerateRoad(const FString& BasePath,const FString& RoadPath, const FRoadParams& RoadParams);

	UFUNCTION(BlueprintCallable, Category = "CityGen")
	void TerrainElevate(const FString& TerrainTifPath, const FTerrainParams& TerrainParams);

	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "CityGen", meta = (CallInEditor=true))
	void OnGenerateFinished(const FString& BasePath);

public:
	/// <summary>
	/// RoadMaterialArray,includes 6 materials, arranged in order
	/// RoadWayMat = 0,
	/// SideWalkMat = 1,
	///	GreenMedianMat = 2,
	///	GreenEdgeMat = 3,
	///	WhiteLineMat = 4,
	///	YellowLineMat = 5,
	/// </summary>
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen")
	TArray<UMaterialInterface*> RoadMaterialArray;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen")
	TArray<UMaterialInterface*> TerrainMaterialArray;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen")
	TArray<UMaterialInterface*> GroundMaterialArray;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CityGen")
	TArray<UMaterialInterface*> WaterMaterialArray;
};
