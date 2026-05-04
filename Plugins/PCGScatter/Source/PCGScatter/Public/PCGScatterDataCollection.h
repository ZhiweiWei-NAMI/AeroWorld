// Copyright (C) Alibaba.inc. DataV, 2024. All Rights Reserved.
#pragma once
#include "CoreMinimal.h"
#include "PCGScatterCommon.h"
#include "Subsystems/WorldSubsystem.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "UObject/ObjectMacros.h"
#include "PCGScatterDataCollection.generated.h"

class AGraphicsPrimitiveProxyBase;
class UPCGScatterDataManager;
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnPCGScatterDataChanged, const FGuid&, DataGuid);
UCLASS(BlueprintType)
class PCGSCATTER_API UPCGScatterDataManager : public UGameInstanceSubsystem
{
public:
	GENERATED_BODY()

	// ~Begin FPCGScatterGeometryBase
	
	/**
	 * 
	 * @param Data 
	 * @return true: 添加成功; false: 集合中已存在相同GUID的数据
	 */
	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	bool AddPCGScatterData(const FPCGScatterData& Data);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	bool DeletePCGScatterData(const FGuid& InGeometryGuid);
	
	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	bool UpdatePCGScatterData(const FGuid& InGeometryGuid,const FPCGScatterData& InPCGScatterData);
	
	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	bool GetPCGScatterData(const FGuid& InGeometryGuid, FPCGScatterData& PCGScatterData);

	template<typename DataType>
	bool GetPCGScatterData(const FGuid& InGeometryGuid, DataType*& GeometryData);
	// ~End FPCGScatterGeometryBase

	// ~Begin Collection

	/**
	 * 创建一个实体要素的数据集合
	 * @param CollectionType 指定集合容纳的数据类型。一个集合只能容纳指定类型的数据 
	 * @return 返回新创建的数据集合的Guid
	 */
	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	FGuid CreateCollection(EPCGScatterType CollectionType);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	bool DeleteCollection(const FGuid& InCollectionGuid);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	bool UpdateCollection(const FGuid& InCollectionGuid, const FPCGScatterDataCollection& InNewPCGScatterCollection);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	FPCGScatterDataCollection& GetCollection(const FGuid& InCollectionGuid);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	bool AddGeometryToCollection(const FGuid& InCollectionGuid, const FPCGScatterData& InPCGScatterData);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	bool DeleteGeometryFromCollection(const FGuid& InCollectionGuid, const FGuid& InPCGScatterDataGuid);
	// ~End Collection

	// ~Begin Group
	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	FGuid CreateGroup();

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	bool DeleteGroup(const FGuid& InGroupGuid);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	bool UpdateGroup(const FGuid& InGroupGuid, const FPCGScatterDataGroup& InPCGScatterGroup);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	FPCGScatterDataGroup& GetGroup(const FGuid& InGroupGuid);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	TArray<FPCGScatterDataGroup> GetAllGroup();

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	bool AddEmptyCollectionToGroup(const FGuid& InGroupGuid);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	bool AddCollectionToGroup(const FGuid& InGroupGuid, const FGuid& InCollectionGuid);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	bool DeleteCollectionFromGroup(const FGuid& InGroupGuid, const FGuid& InCollectionGuid);

	// ~End Group

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	const TMap<FGuid,FPCGScatterData>& GetAllPCGScatterData();

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	void GetTypedPCGScatterData(EPCGScatterType InType, TArray<FPCGScatterData>& OutPCGScatterDataArray);

	// TODO 已有UpdatePCGScatterdata接口，该函数是否还需要
	// UFUNCTION(BlueprintCallable)
	void SetPCGScatterData(const FGuid& InDataGuid, const FPCGScatterData& InPCGScatterData);

	void RemoveAllDataFromRoot();
	
	TArray<FPCGScatterDataSavInfo> ExportPCGScatterDataInternal();

	/**
	 * 导出 PCGScatterData 到文件
	 * @param OutFilePath 导出的 PCGScatterData 的位置，例如 "C:\Users\ABC\Desktop\data.json"
	 */
	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	void ExportPCGScatterData(FString OutFilePath);

	/**
	 * 合并一组点集对象
	 * @param PointDataArray 已有的点集数据对象数组
	 * @param MergedType 将要合并成的数据对象的类型 Line or Polygon
	 * @param bKeepOriginData 是否保留原始的点集对象数据，true=保留原始点集数据；false=删除原始点集数据
	 */
	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	void MergePointData(const TArray<FPCGScatterData>& PointDataArray, EPCGScatterType MergedType, FGuid& OutNewMergedPCGScatterData, bool bKeepOriginData = true);

	/**
	 * 基于已有数据对象，拆分生成一个新的点集对象
	 * @param InGeometryGuid 已有的数据对象 Guid
	 * @param Indices 要抽取出的数据的序号列，例如抽取出当前对象的 [第 3 号，第 5 号，第 1 号] 来生成新的数据对象
	 * @param OutDataGuid 生成的新的点集对象的 Guid
	 */
	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataManager")
	void SplitData2PointData(const FGuid& InGeometryGuid, const TArray<int>& Indices, FGuid& OutGeometryGuid);
	
public:
	
	UPROPERTY()
	FOnPCGScatterDataChanged OnPcgScatterDataChanged;
	
private:
	/** GeometryGuid.Guid --->  FPCGScatterGeometryBase; 一个PointArray只会属于一个Collection */
	UPROPERTY()
	TMap<FGuid,FPCGScatterData> MapOfPCGScatterData;	/** 新结构 */
	
	/** Collection.Guid ---> TArray<FPCGScatterGeometryBase.Guid>; 一个Collection只会属于一个Group */
	// TMap<FGuid, TArray<FGuid>> Collections;
	UPROPERTY()
	TMap<FGuid,FPCGScatterDataCollection> MapOfDataCollections;	/** 新结构 */
	
	/** Group.Guid ---> TArray<Collection.Guid> */
	// TMap<FGuid, TArray<FGuid>> Groups;
	UPROPERTY()
	TMap<FGuid, FPCGScatterDataGroup> Groups;
};

template <typename DataType>
bool UPCGScatterDataManager::GetPCGScatterData(const FGuid& InGeometryGuid, DataType*& GeometryData)
{
	if (!MapOfPCGScatterData.Contains(InGeometryGuid))
	{
		return false;
	}
	GeometryData = Cast<DataType>(MapOfPCGScatterData[InGeometryGuid].GeometryData);
	return true;
}
